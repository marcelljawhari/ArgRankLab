# ArgRankLab

**An empirical correlation analysis of ranking semantics for abstract argumentation frameworks.**

This project implements various ranking semantics for abstract argumentation frameworks (AFs) and conducts an empirical analysis of their correlations. This is the companion code for the Bachelor Thesis by Marcell Jawhari.

## Description

The primary goal of this project is to:
1. Implement selected ranking semantics:
   - Discussion-based Semantics (Dbs)
   - Categoriser-based Semantics (Cat)
   - Probabilistic Semantics (Prob)
   - Serialisability-based Semantics (Ser)
2. Apply these semantics to benchmark AFs from the International Competition on Computational Models of Argumentation (ICCMA).
3. Calculate and analyze the correlation (e.g., using Kendall's Tau and Spearman's Rho) between the rankings produced by these different semantics.
4. Investigate how structural properties of AFs might influence these correlations.

## Setup

1. Clone the repository (if remote is set up):
   `git clone https://github.com/marcelljawhari/ArgRankLab`
2. Navigate to the project directory:
   `cd ArgRankLab`
3. Create a virtual environment (recommended):
   `python -m venv venv`
   `source venv/bin/activate`  # On Windows: `venv\Scripts\activate`
4. Install dependencies:
   `pip install -r requirements.txt`

## License

This project is licensed under the [MIT License](LICENSE).