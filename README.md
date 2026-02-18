#This repository contains code for Shea et al. All methods are in preprocess.py and splicing_ode_model.py.
#If running from this directory, run these commands:
#
#Pre-processing events:
#This command identifies events that satisfy 
#-0.1 < sheaPsi - tannerPsi < 0.3 and where |shea_dPsi (WT - HSALR)} 
#and |tanner_dPsi (WT - HSALR)| > 0.2.:
python3 script.py preprocess.shea_vs_tanner shea_vs_tanner_psi.txt output/shea_vs_tanner.scatters.pdf output/selected_events_psifilters.txt

#This command computes the mean squared error of individual splicing events across 
#animal replicates of the EEV-PMO time course:
python3 script.py preprocess.filter_by_mse psi_with_params.txt output/selected_events_mse.txt output/mse.plot.pdf

#This command selects events that are in both pre-processed outputs.
python3 script.py preprocess.overlap_lists output/selected_events_psifilters.txt output/selected_events_mse.txt output/selected_events.txt

#Running the model:
python3 script.py splicing_ode_model.infer psi_with_params.txt output/selected_events.txt output/model.db 5000 

#Output percentiles for each parameter:
python3 script.py splicing_ode_model.output_percentiles psi_with_params.txt output/selected_events.txt output/model.db output/model_percentiles.txt

#Plotting psi from model:
python3 script.py splicing_ode_model.plot_results_with_psi psi_with_params.txt output/selected_events.txt output/model.db output/model_output_psi.pdf output/model_output_parameters.txt

#Compare to nascent RNAseq measurements:
python3 script.py splicing_ode_model.k_vs_nascent TPM_summary_E.txt psi_with_params.txt output/selected_events.txt output/model.db output/k_vs_nascent.pdf

#Running the model for Tanner et al:
python3 script.py splicing_ode_model.infer_Tanner Tanner_S5_parameters.txt output/Tanner_model.db 500

#Output percentiles for each parameter:
python3 script.py splicing_ode_model.output_percentiles_Tanner Tanner_S5_parameters.txt output/Tanner_model.db output/Tanner_percentiles.txt

#Compare Shea K to Tanner K:
python3 script.py splicing_ode_model.shea_vs_tanner psi_with_params.txt output/selected_events.txt output/model.db Tanner_S5_parameters.txt output/Tanner_model.db output/shea_vs_tanner.k.pdf 



