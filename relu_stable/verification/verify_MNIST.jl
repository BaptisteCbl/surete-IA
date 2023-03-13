using MIPVerify
using JuMP
using Gurobi
using Memento
using MAT

model_name = ARGS[1]
println(model_name)
eps = parse(Float64, ARGS[2])

if isassigned(ARGS, 3)
    start_index = parse(Int64, ARGS[3])
else
    start_index = 1
end
if isassigned(ARGS, 4)
    end_index = parse(Int64, ARGS[4])
else
    end_index = 10000
end

path="./$(model_name)/mat/model.mat"
param_dict = path |> matread

c1_size = 3136
c2_size = 1568
c3_size = 100

fc1 = get_matrix_params(param_dict, "fc1", (784, c1_size))
if haskey(param_dict, "fc1/mask")
    m1 = MaskedReLU(dropdims(param_dict["fc1/mask"]; dims=1), interval_arithmetic)
else
    m1 = ReLU(interval_arithmetic)
end
fc2 = get_matrix_params(param_dict, "fc2", (c1_size, c2_size))
if haskey(param_dict, "fc2/mask")
    m2 = MaskedReLU(dropdims(param_dict["fc2/mask"]; dims=1))
else
    m2 = ReLU()
end
fc3 = get_matrix_params(param_dict, "fc3", (c2_size, c3_size))
if haskey(param_dict, "fc3/mask")
    m3 = MaskedReLU(dropdims(param_dict["fc3/mask"]; dims=1))
else
    m3 = ReLU()
end
softmax = get_matrix_params(param_dict, "softmax", (c3_size, 10))

nnparams = Sequential(
    [Flatten(4), fc1, m1, fc2, m2, fc3, m3, softmax],
    "data0"
)

mnist = read_datasets("MNIST")

f = frac_correct(nnparams, mnist.test, 10000)
println("Fraction correct: $(f)")

# NOT WORKING PROPERLY
# when time limit is reached, solver crash
# not sure if this is due to Gurobi, JuMP or MIPVerify

#println("Verifying $(start_index) through $(end_index)")
#target_indexes = start_index:end_index
#
#MIPVerify.set_log_level!("info")
#
##setting Gurobi solvers
##const GRB_ENV = Gurobi.Env()
##optimizer = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV); add_bridges = false)
##optimizer = JuMP.Model(Gurobi.Optimizer)
##set_optimizer_attribute(optimizer, "TimeLimit", 120)
##set_optimizer_attribute(optimizer, "BestObjStop", eps)
#
##tightening_solver = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV); add_bridges = false)
##set_optimizer_attribute(tightening_solver, "TimeLimit", 5)
##set_optimizer_attribute(tightening_solver, "OutputFlag", 0)
#
#
#d=MIPVerify.batch_find_untargeted_attack(
#    nnparams, 
#    mnist.test, 
#    target_indexes, 
#    Gurobi.Optimizer,
#    Dict("TimeLimit" => 120, "BestObjStop" => eps),
#    save_path="./$(model_name)/verification/results/",
#    solve_rerun_option = MIPVerify.resolve_ambiguous_cases,
#    pp = MIPVerify.LInfNormBoundedPerturbationFamily(eps),
#    norm_order=Inf, 
#    tightening_algorithm=lp,
#    tightening_options=Dict("TimeLimit" => 20)
#)