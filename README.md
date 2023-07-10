# GIS-IDW-Interpolation
An IDW-based approach to interpolation particle pollution data using a Leave-One-Out Cross Validation. Running is (of course) resource intensive [~7.5 hour runtime on RTX 2070/16GB DDR4]. Note: The final outputs created by the given input files are not listed here as they are >500MBs a file. If one would like to verify the project read Final.docx. 


## Implementation
This is the GUI one finds when running the code:

![GUI](/readme1.PNG?raw=true "GUI")

### Import Data
Import Data... imports data. If data is properly imported it will generate a mapping of said data.

![Map](/readme2.PNG?raw=true "Map")

### Perform Interpolation
Performs our IDW-based Interpolation. Refer to the Final_Project.docx for more information.

### Generate Data, Analytics, and Query
The rest of the program is straightforward. We take our interpolated data and perform a Leave-One-Out-Cross Validation on the results and generate results from that:

![data](/readme3.PNG?raw=true "data")

![analytics](/readme4.PNG?raw=true "analytics")
