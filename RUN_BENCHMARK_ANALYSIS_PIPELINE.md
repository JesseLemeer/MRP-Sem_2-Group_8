\# Running the Benchmark Analysis Pipeline



This file explains how to run the notebook:



```text

run\_benchmark\_analysis\_pipeline.ipynb

```



The notebook is used to run the benchmark analysis pipeline for the selected adversarial robustness experiments. It produces the main analysis outputs used in the report, including transferability results, SAT results, CKA summaries, CKA-transferability correlations, and pairwise layer-wise/cross-layer CKA figures.



\## 1. Before running the notebook



Make sure you are inside the project repository:



```bash

cd MRP-Sem\_2-Group\_8

```



Make sure the required Python environment is activated.



On the UM cluster, the environment can be prepared using:



```bash

source /cvmfs/software.eessi.io/versions/2023.06/init/bash

module load Python/3.11.5-GCCcore-13.2.0

source ../xai\_env/bin/activate

```



Check that Python is available:



```bash

python --version

```



\## 2. Select the benchmark



The notebook uses the environment variable `SELECTED\_BENCHMARK` to decide which benchmark to run.



Available benchmark names:



```text

cifar10\_linf

cifar10\_l2

cifar100\_linf

imagenet\_linf

```



Example:



```bash

export SELECTED\_BENCHMARK=cifar10\_linf

```



For PowerShell on Windows, use:



```powershell

$env:SELECTED\_BENCHMARK="cifar10\_linf"

```



\## 3. Run the notebook manually



You can open the notebook in Jupyter and run all cells:



```bash

jupyter notebook run\_benchmark\_analysis\_pipeline.ipynb

```



Then select:



```text

Kernel -> Restart \& Run All

```



\## 4. Run the notebook from the command line



To execute the notebook without opening Jupyter, use:



```bash

jupyter nbconvert --to notebook --execute run\_benchmark\_analysis\_pipeline.ipynb --output executed\_run\_benchmark\_analysis\_pipeline.ipynb --ExecutePreprocessor.timeout=-1

```



For Windows PowerShell, the same command can be used:



```powershell

jupyter nbconvert --to notebook --execute run\_benchmark\_analysis\_pipeline.ipynb --output executed\_run\_benchmark\_analysis\_pipeline.ipynb --ExecutePreprocessor.timeout=-1

```



\## 5. Run all benchmarks



To run all four benchmarks one by one on Linux or the UM cluster:



```bash

for benchmark in cifar10\_linf cifar10\_l2 cifar100\_linf imagenet\_linf

do

&#x20;   export SELECTED\_BENCHMARK=$benchmark

&#x20;   jupyter nbconvert --to notebook --execute run\_benchmark\_analysis\_pipeline.ipynb --output executed\_${benchmark}.ipynb --ExecutePreprocessor.timeout=-1

done

```



For Windows PowerShell:



```powershell

$benchmarks = @("cifar10\_linf", "cifar10\_l2", "cifar100\_linf", "imagenet\_linf")



foreach ($benchmark in $benchmarks) {

&#x20;   $env:SELECTED\_BENCHMARK = $benchmark

&#x20;   jupyter nbconvert --to notebook --execute run\_benchmark\_analysis\_pipeline.ipynb --output "executed\_$benchmark.ipynb" --ExecutePreprocessor.timeout=-1

}

```



\## 6. Main output folders



The notebook saves the generated outputs into benchmark-specific folders. The main outputs include:



```text

saved\_outputs\_cloud/<benchmark>/cloud\_final/

```



Important output types include:



```text

transferability\_matrix.csv

sat\_matrix.csv

layerwise\_cka.csv

cross\_layer\_cka.csv

cka\_group\_correlations.csv

figures/

```



The report-ready files are stored in:



```text

report\_figures/

github\_report\_outputs/

```



The SAT CSV files used for report analysis are stored in:



```text

github\_report\_outputs/sat/<benchmark>/sat\_long.csv

github\_report\_outputs/sat/<benchmark>/sat\_matrix.csv

```



\## 7. Notes



\* The notebook should be run separately for each benchmark.

\* `SELECTED\_BENCHMARK` must be set before execution.

\* The ImageNet benchmark uses the selected 1,000-image validation subset described in the report.

\* The notebook name is intentionally general because it does not only compute CKA. It also supports transferability, SAT, and report-ready analysis outputs.

\* Generated result folders should not be committed unless they are final report outputs needed by the team.



