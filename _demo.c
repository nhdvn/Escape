/* small-scale ESCAPE _demo.c
 * make _demo && ./_demo [seed]
 * make -B _demo DEMO="-DDEMO_N=4 -DDEMO_M=5" */
#ifndef DEMO_N
#define DEMO_N 3
#define DEMO_M 3
#endif
#define PLHE_MC1        DEMO_M
#define BLOCK_SIZE_BITS 8
#include "plhe/plhe.c"
#include <stdlib.h>

enum { n = DEMO_N, m = DEMO_M, t = n * m, N = t * m, LINE = 256 };

typedef struct { int row[n], col[n]; uint8_t rho; } hint_t;   /* global 0-based rows */

static plhe_q_t    dh[2][t][PLHE_LAMBDA];   /* server: Regev hint of each DB (0) / DT (1) row */
static hint_t      hints[64 * N];
static plhe_rng_t  rng;
static int         n_hints;
static const char *VIEW[2] = { "DB", "DT" };

static uint8_t DB(int x) { return (uint8_t)x; }
static int part(int x) { return (x - 1) / (m * m); }
static int a_of(const hint_t *h, int k) { return h->row[k] * m + h->col[k] % m + 1; }

static int at(int view, int r, int j)               /* label of entry j in row r */
{
    return view ? r / m * m * m + j * m + r % m + 1 : r * m + j + 1;
}

static const char *list(const int *v, int add)      /* "[v0+add,v1+add,...]" */
{
    static char buf[4][128];
    static int b;
    char *s = buf[b++ % 4], *p = s;
    for (int k = 0; k < n; k++) p += sprintf(p, "%c%d", k ? ',' : '[', v[k] + add);
    strcpy(p, "]");
    return s;
}

static int holder(int x)                            /* first hint whose A holds x */
{
    int i = 0;
    while (i < n_hints && a_of(&hints[i], part(x)) != x) i++;
    return i;
}

static hint_t random_hint(int x)                    /* holds x if x > 0 */
{
    hint_t h = { .rho = 0 };
    for (int k = 0; k < n; k++) {
        h.row[k] = k * m + rand() % m;
        h.col[k] = k * m + rand() % m;
    }
    if (x > 0) {
        h.row[part(x)] = (x - 1) / m;
        h.col[part(x)] = part(x) * m + (x - 1) % m;
    }
    return h;
}

static void print_db(void)
{
    printf("Example 1: n = %d partitions of %d x %d, N = %d entries\n\n", n, m, m, N);
    printf("  %-*s      DT\n", 4 * t + 2 * (n - 1), "DB");
    for (int line = -1; line < m; line++) {         /* row numbers, then entries */
        for (int view = 0; view < 2; view++) {
            printf(view ? "      " : "  ");
            for (int r = 0; r < t; r++)
                printf("%s%4d", r % m == 0 && r ? " |" : "", line < 0 ? r + 1 : at(view, r, line));
        }
        printf("\n");
    }
}

static void print_hints(int mark)                   /* the refreshed hint marked with * */
{
    char id[16], cell[LINE];
    printf("\n[hint table]\n");
    for (int i = 0; i < n_hints; i++) {
        snprintf(id, sizeof id, "h%d:", i + 1);
        snprintf(cell, LINE, "%-4s %s ∩ %s", id, list(hints[i].row, 1), list(hints[i].col, 1));
        printf("%s%c%-*s", i % 4 ? "   " : " ", i == mark ? '*' : ' ', 12 + 6 * n, cell);
        if (i % 4 == 3 || i == n_hints - 1) printf("\n");
    }
}

static void offline(void)
{
    uint8_t v[m];
    printf("\n[offline server]: compute Regev hint\n");
    for (int view = 0; view < 2; view++)
        for (int r = 0; r < t; r++) {
            for (int j = 0; j < m; j++) v[j] = DB(at(view, r, j));
            plhe_dhint_row(dh[view][r], v, view * t + r);
        }
    for (int x = 1; x <= N; x++)                    /* until every entry is in a hint */
        while (holder(x) == n_hints) hints[n_hints++] = random_hint(0);

    printf("[offline client]: database streaming\n");
    for (int x = 1; x <= N; x++)
        for (int i = 0; i < n_hints; i++)
            if (a_of(&hints[i], part(x)) == x) hints[i].rho += DB(x);
}

/* reveal one row per partition and fetch the parity of A \ {x} */
static uint8_t fetch(int view, const hint_t *h, int i, int x, const char *title)
{
    const int *pick = view ? h->col : h->row, *other = view ? h->row : h->col;
    int kx = part(x), rows[n], a[n];
    char bits[n + 1] = "";
    uint8_t key[PLHE_LAMBDA];
    plhe_q_t ct[n][m], gamma = 0, dstar[PLHE_LAMBDA] = { 0 };

    for (int k = 0; k < n; k++) {
        rows[k] = pick[k];
        a[k] = a_of(h, k);
        bits[k] = k == kx ? '0' : '1';              /* selection vector */
    }
    if (rand() % m)                                 /* keep row with prob 1/m */
        rows[kx] = kx * m + (pick[kx] % m + 1 + rand() % (m - 1)) % m;
    
    printf("\n%s hint h%d with A=%s holds x=%d\n", title, i + 1, list(a, 0), x);
    printf("  replace %s row %d with %d (%s)\n", VIEW[view], pick[kx] + 1, rows[kx] + 1,
           rows[kx] == pick[kx] ? "keep" : "swap");
    printf("  -> reveal %s rows = %s with %s\n", VIEW[view], list(rows, 1), bits);

    plhe_keygen(key, &rng);
    for (int k = 0; k < n; k++) {                   /* client */
        plhe_encrypt_vbit(ct[k], view * t + rows[k], other[k] % m, key, &rng);
        if (k == kx) ct[k][other[k] % m] -= PLHE_DELTA;
    }
    for (int k = 0; k < n; k++) {                  /* server */
        for (int j = 0; j < m; j++) gamma += DB(at(view, rows[k], j)) * ct[k][j];
        for (int l = 0; l < PLHE_LAMBDA; l++) dstar[l] += dh[view][rows[k]][l];
    }
    for (int l = 0; l < PLHE_LAMBDA; l++)           /* client */
        gamma -= key[l] * dstar[l];
    uint8_t parity = (gamma + PLHE_DELTA / 2) >> (PLHE_Q_BITS - PLHE_P_BITS);

    printf("  client decrypts rho' =");
    for (int k = 0; k < n; k++) printf("%s %d*%d", k ? " +" : "", k != kx, DB(a[k]));
    printf(" = %d\n", parity);
    return parity;
}

static void query_and_refresh(int x)
{
    int i = holder(x);
    if (i == n_hints) {
        printf("no hint holds x=%d; try another\n", x);
        return;
    }
    uint8_t rho_q = fetch(1, &hints[i], i, x, "[query] use");
    uint8_t beta = hints[i].rho - rho_q;
    printf("  -> DB[%d] = rho - rho' = %d - %d = %d  (%s)\n", x, hints[i].rho, rho_q, beta,
           beta == DB(x) ? "correct" : "WRONG");

    hint_t fresh = random_hint(x);
    uint8_t rho_r = fetch(0, &fresh, i, x, "[refresh] new");
    fresh.rho = rho_r + beta;
    hints[i] = fresh;
    printf("  -> new rho = rho' + DB[%d] = %d + %d = %d\n", x, rho_r, beta, fresh.rho);
    print_hints(i);
}

int main(int argc, char **argv)
{
    unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 1;
    srand(seed); plhe_rng_init(&rng, seed);
    
    char line[64];
    print_db(); offline(); print_hints(-1);
    
    while (1) {
        printf("\nindex x in [1, %d] to retrieve (q to quit)> ", N);
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin) || line[0] == 'q') break;
        int x = atoi(line);
        if (x >= 1 && x <= N) query_and_refresh(x);
    }
    return 0;
}
