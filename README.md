## Manifold Embedded Knowledge Transfer for Brain-Computer Interfaces

This repository contains code of manifold embedded knowledge transfer (MEKT). MEKT is a novel transfer learning framework for offline unsupervised cross-subject electroencephalogram (EEG) classification. It can cope with variations among different individuals and/or tasks in unsupervised scenarios, and has achieved superior performance.

<div align="center">
    <img src="presentation/diagram.png", width="750">
</div>

The average BCAs on the four datasets are shown in Table II. All MEKT-based approaches outperformed the baselines. 

<div align="center">
    <img src="presentation/bca.png", width="365">
</div>

The results on STS/MTS tasks above demonstrates the effectiveness of our approach.

## Running the code
The code is MATLAB code works in Windows 10 system.

Code files introduction:

**demo_mi1_sts.m** -- Single-source to Single-target (STS) tasks on MI dataset 1, run this demo file in MATLAB could show the performance similar to our paper. It contains the dataset test, MEKT approach function, and DTE test sections.

**demo_mi2_mts.m** -- Multi-source to Single-target (MTS) tasks on MI2 dataset.

**demo_rsvp_sts.m** -- STS tasks on RSVP dataset. The augmented covariance matrix is used here.

**demo_ern_mts.m** -- MTS tasks on ERN dataset. The augmented covariance matrix is used here.

**dte_rsvp.m** -- Domain transferability estimation (DTE) to select source domians test on RSVP dataset.

**MEKT.m** -- It's the implementation of MEKT approach. Please find the specific input/output instructions in the function comments.