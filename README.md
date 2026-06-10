# MRP-Sem_2-Group_8

The run_benchmark_analysis_pipeline notebook includes all selected model configurations in a single configuration list, which makes the code easier to control and extend. Each model entry contains the model name, RobustBench identifier, dataset, threat model, perturbation budget, architecture, and training type. This makes it possible to add or disable models without changing the rest of the pipeline.

The notebook is also designed to run the benchmark groups separately. This is controlled by the variable "SELECTED_BENCHMARK". For example, when this variable is set to cifar10_linf, the pipeline only loads and evaluates the models that belong to the CIFAR-10 L_inf benchmark group. This separation keeps the experiment organised and prevents incompatible models from being compared together.

To support execution on the UM GPU cluster, a batch script was added: run_all_benchmarks_cloud.sh 

Although the script name contains the word cloud, in this project it was used to run the experiments on the UM GPU infrastructure. The script automates the execution of the main notebook for each benchmark group.

The command used on the UM GPU cluster was: bash run_all_benchmarks_cloud.sh 

The script runs the notebook separately for each benchmark group and saves the outputs in separate folders under: saved_outputs_cloud/<benchmark>/<run_mode>/

Each benchmark folder contains its own transferability, CKA, their correlation, SAT, and figure outputs. This prevents results from different datasets or threat models from being mixed or overwritten.

The implementation has different running modes. A debug mode is used first to test the full pipeline with fewer examples and lower computational cost. After confirming that the pipeline runs correctly, the cloud final mode can be used for the full experiment. This mode uses AutoAttack and a larger number of examples, which gives more reliable final results. A maximum cloud mode is also available for stronger GPU servers, but it should only be used if the available memory and running time are sufficient.
