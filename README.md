# Deep neural networks to predict foraging behaviour: salt-water immersion data can accurately predict diving in seabirds

This directory contains code and directory structure for the research paper "*Deep neural networks to predict foraging behaviour: salt-water immersion data can accurately predict diving in seabirds*." 

This originally started as an MRes research project by Luke Swaby's at the [Institute of Zoology](https://www.zsl.org/what-we-do/institute-of-zoology) for the [MSc Computational Methods in Ecology and Evolution](https://www.imperial.ac.uk/study/pg/life-sciences/computational-methods-ecology-evolution/) course at Imperial College London

## Prerequisites

This project was developed on a Unix OS.

The following programming languages/applications are used in the project:
* R (4.1.0)
* Python (3.9.6)
* Jupyter-Notebook (6.1.4)

## Dependencies

### R
* `caret` (6.0.88)
* `data.table` (1.14.0) 
* `dplyr` (1.0.7)
* `GeoLight` (2.0.0)
* `geosphere` (1.5.10)
* `ggplot2` (3.3.5)
* `ggspatial` (1.1.5)
* `ggrepel` (0.9.1)
* `gridExtra` (2.3)
* `grid` (4.1.0)
* `plyr` (1.8.6)
* `rnaturalearth` (0.1.0)
* `sf` (1.0.0)
* `sp` (1.4.5)
* `stringr` (1.4.0)
* `splitstackshape` (1.4.8)
* `tools` (4.1.0)
* `zoo` (1.8.9)

### Python
* `dask` (2021.06.2)
* `numpy` (1.19.5)
* `pandas` (1.2.5)
* `sklearn` (0.24.2)
* `tensorflow` (2.5.0)

### Jupyter Notebook
* R kernel (install [here](https://github.com/IRkernel/IRkernel))

## Structure and Usage

This directory contains the following folders:
* **Data**: empty directory to store contents of data file found at ...
* **Code**: contains all code scripts for the project. Descriptions of script functionality can be found at the top of each script (or in the help file in the case of Python scripts).
* **Results**: empty directory for results files to be pushed into.
* **Plots**: empty directory for plots to be pushed into.

Once data files have been added, the project directory structure should look like this:
```
PROJECT
│   README.md
│
└───Data/
│   │   
│   └───BIOT_DGBP/
│   │    │   ...
│   │    └───BIOT_DGBP/
│   │         │   ...
│   │
│   └─── GLS Data 2019 Jan DG RFB Short-term/
│        │   ...
│        └───matched/
│             │   ...
└───Code/
│   │   ...
│
└───Results/
│   │   NA
│
└───Plots/
    │   NA
```

To run the full pipeline, navigate to the `Code/` directory and run the following command:
```
$ sh RUN_PROJECT.sh
```

## Contact

Email: <lds20@ic.ac.uk>.
