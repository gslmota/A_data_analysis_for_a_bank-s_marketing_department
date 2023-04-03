# A_data_analysis_for_a_bank-s_marketing_department
This analysis consists of identifying the main customers and the best products in relation to a bank's revenue. These analyzes will be passed on to the marketing department for the creation of personalized products for these clients.

# Contents
``` shell
.
├── codes
│   ├── Análise_para_o_departamento_de_marketing_de_um_banco.ipynb
│   └── análise_para_o_departamento_de_marketing_de_um_banco.py
├── imgs
│   ├── elb.png
│   ├── pca2.png
│   ├── pcaAu.png
│   └── pca.png
├── LICENSE
└── README.md
```
The **codes** folder contains the python codes used in projects.The other folders follow the same pattern.

# Requirements

 * Check the **requirements.txt** file for libs.


# Test

```shell
git clone https://github.com/gslmota/A_data_analysis_for_a_bank-s_marketing_department.git
cd A_data_analysis_for_a_bank-s_marketing_department
pip install -r requirements.txt
```



# Results

In this project, a comparison of the simple use of the Elbow Method to identify clusters was carried out, and then the PCA was applied to the data grouped by Kmeans. Another approach could have used Autoencoders instead of applying PCA for dimensionality reduction. However, it was applied as a way of grouping data that had high correlation in pre-processing. To later forward this data to Elbow, Kmeans and PCA.

### **Elbow->Kmeans->PCA**: 


* Elbow Clusters

![!imgs](https://github.com/gslmota/A_data_analysis_for_a_bank-s_marketing_department/blob/main/imgs/elb.png)

* Model Clusters Division

![!imgs](https://github.com/gslmota/A_data_analysis_for_a_bank-s_marketing_department/blob/main/imgs/pca.png)


### **AutoEncoders->Elbow->Kmeans->PCA**: 
This project using Radom Forest had 80% acuracy.

* Elbow Clusters with autoencoders preprocessing vs Simple Elbow

![!imgs](https://github.com/gslmota/A_data_analysis_for_a_bank-s_marketing_department/blob/main/imgs/pcaAu.png)

* Model Clusters Division

![!imgs](https://github.com/gslmota/A_data_analysis_for_a_bank-s_marketing_department/blob/main/imgs/pca2.png)

# References:
* IA Expert Academy: [IA Expert Academy](https://iaexpert.academy/)


