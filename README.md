# Food Quality Inspection & Shelf-Life Risk Prediction System

## A Real-World Deep Learning Product Built With a Deployment-First Mindset

---
<img width="1886" height="986" alt="image" src="https://github.com/user-attachments/assets/0b62e7fa-c454-4712-a9a1-048b2b6abac4" />

## Why I Built This Project

I intentionally chose to build a **small-scope but real-world system** instead of a large academic or research-only project.

As a Data Scientist, I realized early that:
- A highly accurate model alone does not solve real problems
- Users do not think in probabilities
- Business impact comes from **clear decisions**, not complex outputs

This project was motivated by a simple question:

> Can I build a system where deep learning actually helps people take better daily decisions?

Instead of chasing scale, I focused on **depth, usability, and deployment readiness**.

---

## The Core Problem This System Solves

In real life:
- Customers are unsure whether food is safe
- Shopkeepers rely on guesswork for batch inventory decisions
- Food wastage increases due to inconsistent quality checks

This system provides **decision support**, not just predictions.

It answers:
- Is the food item fresh or risky?
- Should it be sold, discounted, or discarded?
- How will freshness change over the next two days?

---

## Who This System Is For

### Customer (Mobile-First Usage)

A customer typically:
- Uses a mobile phone
- Uploads an image or captures one using the camera
- Wants a simple, confident answer

Customer gets:
- Freshness score (0–100)
- Risk level (Low / Medium / High)
- Clear safety comment
- Shelf-life trend for the next 2 days

This helps customers make **confident consumption decisions**.

---

### Shopkeeper (Batch & Operational Usage)

A shopkeeper:
- Inspects food in batches
- Works under time pressure
- Needs fast, defensible decisions

Shopkeeper can:
- Upload multiple images together (folder-style batch)
- Use camera to scan items
- Receive:
  - Item-wise risk analysis
  - Batch-level summary
  - Clear inventory action
  - Downloadable CSV report

This directly impacts:
- Inventory planning
- Waste reduction
- Discount and pricing decisions

---

## Model & Machine Learning Approach

This system uses a **Convolutional Neural Network (CNN)** for image-based classification.

Key points:
- CNN architecture based on **MobileNetV2**
- Trained on the Kaggle dataset  
  *“Fruits Fresh and Rotten for Classification”*
- Binary classification:
  - Fresh
  - Rotten
- Input size: 224 × 224 × 3
- Optimized for:
  - Stability
  - Fast inference
  - Real-world usability

As a Data Scientist, **model training, validation, evaluation, and robustness** were fully owned and designed by me.

---

## From Model Output to Business Decision

The system does not expose raw probabilities.

Instead:
1. Model predicts probability of spoilage
2. Probability is converted into a **Freshness Score**
3. Freshness score is mapped to:
   - Risk level
   - Business action
   - Human-readable comment
4. A rule-based decay estimates freshness for:
   - Today
   - Tomorrow
   - Day after tomorrow

This keeps the system **simple, interpretable, and useful**.

---

## Application Layer & Deployment

The application is built using **Streamlit** and structured like a real internal tool.

Important transparency note:

> As a Data Scientist, my primary focus and ownership was on model development, decision logic, and system design.  
> For the application integration (`app.py`), I took **partial assistance for UI structuring and component wiring**, while fully understanding and validating every logic layer.

This reflects **real industry collaboration**, where:
- Data Scientists focus on intelligence
- UI and integration are often collaborative

The system is fully runnable locally and deployment-ready.

---

## How to Run the Application Locally

### Requirements
- Python 3.9+
- TensorFlow
- Streamlit
- NumPy, Pandas, Matplotlib

### Run Command
```bash
streamlit run app.py
