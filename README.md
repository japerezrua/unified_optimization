# A code for unified optimization of wind farms in terms of AEP from Wind Farm Layout and Cable Routing Collection System
In addition to the unified optimization, i.e., mathematically fully couple both wind farm and cable routing optimization in a monolithic extensive formulation, a code for sequential optimization is also available. 
See below a very brief guide of the content of this repo: 
- The folder <em>unified</em> contains the set of scripts to run the unified optimization approach while relaxing the cable crossing constraints.
- The folder <em>post_proc_unified</em> contains the set of scripts to run the output .dill file from <em>unified</em> and provides the 5 top-performing solutions after forcing the cable crossing constraints.
- The folder <em>sequential</em> contains the set of scripts to run the sequential optimization approach, where first the wind farm layuout is solved, followed up by the cable routing.

See diagram below:


***Important note***
Apart from standard python libraries, CPLEX python API is required to be installed to use these scripts. Find a guide on how to download the full academic version at https://community.ibm.com/community/user/ai-datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students.
