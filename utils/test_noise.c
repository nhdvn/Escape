/* test_noise.c -- Escape v1 noise budget test.
 *
 * Validates the noise math that compress_response relies on: the LWE answer
 * `(agg[c], ans[c])` must homomorphically decrypt to a clean
 * `Delta*full_sum(c) + noise` with |noise| < Delta/2 for the rounded recovery to
 * land on the right plaintext bits.
 *
 * v1 itself does NOT have a plaintext cancellation on the client -- that
 * step happens inside compress_response (server side, encrypted under
 * Paillier; the client just decrypts the final Paillier ciphertext). This
 * test mirrors the underlying LWE math in the clear, so we can measure
 * the noise distribution that compress_response will inherit.
 *
 * Model (per query, one chunk c, summed over k partitions):
 *
 *   ct_k[j]       = <A[k,r_k][j,:], sk> + e_j + Delta*[j == c_k]
 *   ans[c]      = Sum_k Sum_j DB[k][r_k][j,c] * ct_k[j]              (server)
 *   agg[c] = Sum_k Sum_j DB[k][r_k][j,c] * A[k,r_k][j,:]        (server)
 *
 *   Homomorphic LWE-decrypt inside compress_response:
 *     ans[c] - <agg[c], sk>
 *       = Sum_k Sum_j DB[k][r_k][j,c] * e_j  +  Delta * Sum_k DB[k][r_k][c_k,c]
 *       =                noise(c)        +     Delta * full_sum(c)
 *
 *   result[c] = round(decrypted / Delta) mod p   => correct iff |noise| < Delta/2.
 *
 * Theoretical std: sigma_noise(c) ~= p * sigma_e * sqrt(N * MC).
 * Scaled-down params for speed; real Delta/p ratio preserved (P=8, Q=32, sigma=3.2).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef uint32_t u32;
typedef uint64_t u64;
typedef int64_t  i64;
typedef uint8_t  u8;

/* -- Scaled-down params (mirror v1's plhe/params.h ratios) ----------------- */
#define T_N        16    /* partitions (PLHE_N1)      */
#define T_MR       16    /* rows / partition          */
#define T_MC       64    /* cols / partition          */
#define T_LAM      128   /* LWE secret dimension      */
#define T_PBITS    8
#define T_QBITS    32
#define T_DELTA    ((u32)1 << (T_QBITS - T_PBITS))
#define T_PMASK    (((u32)1 << T_PBITS) - 1u)
#define T_SIG_E    3.2

/* -- RNG and Gaussian sampler (xorshift + Box-Muller) --------------------- */
static u64 rng_state;
static void rng_seed(u64 s) { rng_state = s ? s : 0xdeadbeefULL; }
static u64  rng64(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static i64 gaussian(double sigma)
{
    double u1 = (rng64() >> 11) * (1.0 / (1ULL << 53));
    double u2 = (rng64() >> 11) * (1.0 / (1ULL << 53));
    return (i64)round(sigma * sqrt(-2.0 * log(u1 + 1e-300)) * cos(2.0 * M_PI * u2));
}

/* -- Test arrays ---------------------------------------------------------- */
static u32 A    [T_N][T_MR][T_MC][T_LAM];   /* per-row matrix A[k,r] */
static u32 DB   [T_N][T_MR][T_MC];          /* one chunk's worth (c=0)         */
static u32 dh   [T_N][T_MR][T_LAM];         /* dhint[k,r] = Sum_j A[k,r,j,:] * DB[k,r,j] */
static u8  sk   [T_LAM];                     /* shared across partitions       */

static void setup(void)
{
    /* Public matrix A (per (k,r,col); fully random). */
    for (int k = 0; k < T_N; k++)
        for (int r = 0; r < T_MR; r++)
            for (int j = 0; j < T_MC; j++)
                for (int l = 0; l < T_LAM; l++)
                    A[k][r][j][l] = (u32)rng64();

    /* DB entries in [0, p). */
    for (int k = 0; k < T_N; k++)
        for (int r = 0; r < T_MR; r++)
            for (int j = 0; j < T_MC; j++)
                DB[k][r][j] = (u32)rng64() & T_PMASK;

    /* Pre-aggregate dhint. */
    for (int k = 0; k < T_N; k++) {
        for (int r = 0; r < T_MR; r++) {
            memset(dh[k][r], 0, sizeof(dh[k][r]));
            for (int j = 0; j < T_MC; j++)
                for (int l = 0; l < T_LAM; l++)
                    dh[k][r][l] += DB[k][r][j] * A[k][r][j][l];
        }
    }

    /* Shared binary secret (matches v1 compress_key). */
    for (int l = 0; l < T_LAM; l++) sk[l] = (u8)(rng64() & 1);
}

typedef struct {
    int  pass;        /* 1 if result == expected for all trials in run */
    u32  abs_noise;   /* |noise| in this run (the residue mod 2^32, sign-corrected) */
} run_result_t;

/* One trial. r_k[k], c_k[k] supplied by caller. */
static run_result_t run_one(const int *r_k, const int *c_k)
{
    /* Encrypt: ct_k[j] = <A[k,r_k][j,:],sk> + e_j + Delta*[j==c_k]. */
    static u32 ct[T_N][T_MC];
    for (int k = 0; k < T_N; k++) {
        for (int j = 0; j < T_MC; j++) {
            u32 dot = 0;
            for (int l = 0; l < T_LAM; l++)
                dot += A[k][r_k[k]][j][l] * (u32)sk[l];
            i64 e = gaussian(T_SIG_E);
            u32 plain = (j == c_k[k]) ? T_DELTA : 0;
            ct[k][j] = dot + (u32)e + plain;
        }
    }

    /* Server-side LWE answer (b, a):
     *   ans      = Sum_k Sum_j DB[k,r_k][j] * ct_k[j]
     *   agg = Sum_k Sum_j DB[k,r_k][j] * A[k,r_k][j,:]   (folded into dh[k][r_k]) */
    u32 ans = 0;
    for (int k = 0; k < T_N; k++)
        for (int j = 0; j < T_MC; j++)
            ans += DB[k][r_k[k]][j] * ct[k][j];

    /* Inside compress_response (homomorphic, encrypted under Paillier in real v1):
     *   decrypted = ans - <agg, sk>
     * Modeled here in the clear so we can read out the noise. */
    u32 agg_dot_sk = 0;
    for (int k = 0; k < T_N; k++)
        for (int l = 0; l < T_LAM; l++)
            agg_dot_sk += dh[k][r_k[k]][l] * (u32)sk[l];

    u32 decrypted = ans - agg_dot_sk;
    u32 result = ((decrypted + T_DELTA / 2) >> (T_QBITS - T_PBITS)) & T_PMASK;

    /* Expected = Sum_k DB[k,r_k][c_k] mod p. */
    u32 expected = 0;
    for (int k = 0; k < T_N; k++) expected += DB[k][r_k[k]][c_k[k]];
    expected &= T_PMASK;

    /* Pre-mask sum for noise calc: decrypted - Delta*full_sum should be ~= noise. */
    u32 full = 0;
    for (int k = 0; k < T_N; k++) full += DB[k][r_k[k]][c_k[k]];
    u32 residue = decrypted - T_DELTA * full;
    /* residue is u32 with wrap; recover signed magnitude. */
    i64 signed_n = (residue & 0x80000000u)
                 ? (i64)residue - (i64)0x100000000LL
                 : (i64)residue;
    u32 abs_noise = (u32)(signed_n < 0 ? -signed_n : signed_n);

    return (run_result_t){ result == expected, abs_noise };
}

int main(int argc, char **argv)
{
    int n_trials = (argc > 1) ? atoi(argv[1]) : 200;

    printf("==== v1 noise budget ====\n");
    printf("  params: N=%d  MR=%d  MC=%d  LAM=%d  P_BITS=%d  Q_BITS=%d  sigma=%.1f\n",
           T_N, T_MR, T_MC, T_LAM, T_PBITS, T_QBITS, T_SIG_E);

    /* Theoretical std: sigma_noise = p * sigma_e * sqrt(N * MC). */
    double pred_std = (double)((u64)1 << T_PBITS) * T_SIG_E
                      * sqrt((double)T_N * (double)T_MC);
    /* Real-config equivalent (for context): N1=256, MC1=256, LAMBDA=1024. */
    double real_std = (double)((u64)1 << T_PBITS) * T_SIG_E
                      * sqrt(256.0 * 256.0);
    printf("  noise sigma: test ~= 2^%.1f   real (N1=MC1=256) ~= 2^%.1f   budget Delta/2 = 2^%d\n",
           log2(pred_std), log2(real_std), T_QBITS - T_PBITS - 1);

    rng_seed(0xC0FFEEULL);
    setup();

    int fails = 0;
    u64 sum_noise = 0;
    u32 max_noise = 0;
    for (int t = 0; t < n_trials; t++) {
        int r_k[T_N], c_k[T_N];
        for (int k = 0; k < T_N; k++) {
            r_k[k] = (int)(rng64() % T_MR);
            c_k[k] = (int)(rng64() % T_MC);
        }
        run_result_t r = run_one(r_k, c_k);
        if (!r.pass) fails++;
        if (r.abs_noise > max_noise) max_noise = r.abs_noise;
        sum_noise += r.abs_noise;
    }
    double avg_noise = (double)sum_noise / n_trials;
    printf("  trials  : %d\n", n_trials);
    printf("  failures: %d\n", fails);
    printf("  max |noise|: %u (= 2^%.1f)   avg |noise|: %.0f (= 2^%.1f)\n",
           max_noise, log2((double)max_noise + 1.0),
           avg_noise, log2(avg_noise + 1.0));
    printf("  margin/Delta: max %.6f   (PASS iff < 0.5)\n",
           (double)max_noise / (double)T_DELTA);

    int ok = (fails == 0);
    printf("\nNOISE: %s (%d failures)\n", ok ? "PASS" : "FAIL", fails);
    return ok ? 0 : 1;
}
