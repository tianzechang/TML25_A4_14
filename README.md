# TML25_A4_XX

**Team Number:** 14
**Course:** Trustworthy Machine Learning (SS 2025)  
**Assignment:** #4 - Explainability  

---

## **Repository Structure**

```
.
├── task1.py      # Task 1 main code logic (Network Dissection)
├── task2.py      # Task 2 main code logic (Grad-CAM)
├── task3.py      # Task 3 main code logic (LIME)
├── task1.pdf     # Task 1 report
├── task2.pdf     # Task 2 report
├── task3.pdf     # Task 3 report
├── task5.pdf     # Task 5 report (Grad-CAM vs LIME comparison)
└── README.md     # This file
```

---

## **Task Descriptions**

### **Task 1: Network Dissection**
- **Report:** `task1.pdf`  
- **Main Code:** `task1.py`  
  - This script demonstrates the analysis of the last 3 layers of ResNet18 trained on ImageNet and Places365, labeling neurons and generating findings.

### **Task 2: Grad-CAM**
- **Report:** `task2.pdf`  
- **Main Code:** `task2.py`  
  - Generates Grad-CAM, AblationCAM, and ScoreCAM visualizations for the 10 provided ImageNet images.

### **Task 3: LIME**
- **Report:** `task3.pdf`  
- **Main Code:** `task3.py`  
  - Visualizes the important regions of each of the 10 images using LIME and compares them with Grad-CAM results.

### **Task 4: LIME Parameters**
- **Status:** Completed and submitted to the server.  
  - The pickle file and token were submitted as required.

### **Task 5: Grad-CAM vs LIME Comparison**
- **Report:** `task5.pdf`  
  - Contains the analysis and IoU-based comparison between Grad-CAM and LIME methods.


---

## **How to Run**
Each task can be executed by running the corresponding `.py` file. Example:
```bash
python task1.py
python task2.py
python task3.py
```
**Note:** Due to large model and dataset requirements (ImageNet, Places365), the repository only includes core logic, not full runnable pipelines.

---

## **Reports**
- All analysis and visualizations are provided in the respective task PDF files (`task1.pdf`, `task2.pdf`, `task3.pdf`, `task5.pdf`).

---

## **Remarks**
- Task 4 submission was completed on the server (pickle parameters submitted).  
- This repository focuses on key code logic and detailed reports for evaluation.
