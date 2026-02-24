# DDI Dataset Codebase
Code for loading DDI data and the models from our paper:<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;***Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set***

For more information, please visit our project page [here](https://ddi-dataset.github.io/) and read our paper [here](https://www.science.org/doi/full/10.1126/sciadv.abq6147).

Our models can be downloaded [here](https://drive.google.com/drive/folders/1oQ53WH_Tp6rcLZjRp_-UBOQcMl-b1kkP) or through the provided code.

The DDI dataset is hosted on Azure. You get a **time-limited download link** (SAS URL) from the [Stanford AIMI dataset page](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965). That link cannot be opened directly in a browser; use one of the methods below to download the entire dataset.

### Option 1: AzCopy (command line)

1. **Install AzCopy**  
   - Windows: [Download AzCopy v10](https://aka.ms/downloadazcopy-v10-windows) (zip), extract, and optionally add the folder to your PATH.  
   - Or install via winget: `winget install Microsoft.AzCopy`

2. **Download the dataset**  
   Replace `YOUR_SAS_URL` with the full SAS URL from Stanford AIMI (the one that starts with `https://aimistanforddatasets01.blob...` and includes `?sv=...&sig=...`).  
   Replace `DDI` with the folder where you want the data (e.g. `C:\Data\DDI`).

   ```bash
   azcopy copy "YOUR_SAS_URL" "DDI" --recursive
   ```

   Example (use your own link; this one expires):

   ```bash
   azcopy copy "https://aimistanforddatasets01.blob.core.windows.net/ddidiversedermatologyimages?sv=2019-02-02&sr=c&sig=..." "DDI" --recursive
   ```

3. **Match the expected layout**  
   The code expects:
   - `DDI/ddi_metadata.csv`
   - `DDI/images/` with all `.png` files (e.g. `000001.png`, `000002.png`, ...)  
   If the Azure container has a different structure, move the CSV and images into this layout after download.

### Option 2: Azure Storage Explorer (GUI)

1. **Install** [Azure Storage Explorer](https://azure.microsoft.com/products/storage/storage-explorer/).

2. **Connect with the SAS URL**  
   - Open Storage Explorer → **Connect** (plug icon) → **Blob container or directory** → **Shared access signature URL (SAS)**.  
   - Paste the full SAS URL from Stanford AIMI → **Next** → **Connect**.

3. **Download**  
   - Open the container, select the folder or all blobs, then **Download** and choose your local folder (e.g. `DDI`).

4. **Match the expected layout**  
   Same as above: ensure `ddi_metadata.csv` and an `images/` folder with all `.png` files are under your `DDI` directory.

> **Note:** The SAS link expires (often after a few weeks). If you get authentication or expiry errors, request a new download link from the Stanford AIMI dataset page.


## Description 
We include code to download and load our models (`ddi_model.py`), load the DDI dataset (`ddi_dataset.py`), evaluate our models on the DDI dataset (`eval_ddi.py`) as well as evaluate our models on an arbitrary dataset  (`eval_data.py`). For `eval_ddi.py` and `eval_data.py`, we provide a command line interface with the following arguments:
- `model_dir`: File path for where to save models.
- `model`: Name of the model to load (HAM10000, DeepDerm, GroupDRO, CORAL, or CDANN).
- `no_download`: Set to disable downloading models.
- `data_dir`: Folder containing dataset to load. In `eval_ddi.py`, `data_dir` should be the root directory and contain (1) a subfolder called `images` containing all the DDI images and (2) a CSV file called `ddi_metadata.csv`. In `eval_data.py`, the structure should match the root directory in [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder) with 2 classes: benign (class 0) and malignant (class 1).
- `eval_dir`: Folder to store evaluation results.
- `use_gpu`: Set to use GPU for evaluation.
- `plot`: Set to show ROC plot.


### Example usage
- Evaluate `DeepDerm` model on the DDI dataset. Data (not included in this repo) is stored in the `DDI` directory, and results will be saved in the `DDI-results` directory.
```bash
>>>python3 eval_ddi.py --model=DeepDerm --data_dir=DDI --eval_dir=DDI-results 
```
- Evaluate `DeepDerm` model on your own dataset (must be annotated as benign/malignant). Data (not included in this repo) is stored in the `MyData` directory, and results will be saved in the `DDI-results` directory.
```bash
>>>python3 eval_data.py --model=DeepDerm --data_dir=MyData --eval_dir=DDI-results 
```


## Citation
If you find this code useful or use the DDI dataset in your research, please cite:
```
@article{daneshjou2022disparities,
  title={Disparities in dermatology AI performance on a diverse, curated clinical image set},
  author={Daneshjou, Roxana and Vodrahalli, Kailas and Novoa, Roberto A and Jenkins, Melissa and Liang, Weixin and Rotemberg, Veronica and Ko, Justin and Swetter, Susan M and Bailey, Elizabeth E and Gevaert, Olivier and others},
  journal={Science advances},
  volume={8},
  number={31},
  pages={eabq6147},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```

