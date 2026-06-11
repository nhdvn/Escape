from estimator import *

# Your LWE parameters with `m` aggregations
params = LWE.Parameters(
    n=1024,
    q=2**32,                      
    Xs=ND.UniformMod(2),          
    Xe=ND.DiscreteGaussian(3.2),  
    m=2**24
)

print(f"--- Matrix LWE Estimate ---")
print(f"n: {params.n}")
print(f"m: {params.m}")
print(f"q: 2^32")
print(f"----------------------------------\n")

# Run the estimator
res = LWE.estimate(params)

for attack, data in res.items():
    print(f"{attack}: {data}")