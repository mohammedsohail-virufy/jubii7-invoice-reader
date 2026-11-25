# Prompts used (kept visible as the brief asks)

1. **Extraction:** "Here is a photo of a bilingual tax invoice. Extract every field into JSON. Keep numbers exactly as printed, including Arabic-Indic digits, and don't correct anything. Mark any field you can't read clearly."
   *Check:* I compared every number to the photo by eye and kept Arabic digits as strings so the checker sees the raw data.

2. **Checker:** "Write a Python checker that recomputes line amounts from delivered qty × unit price, then subtotals, the 2% Site-B discount, the taxable amount, 5% VAT on the taxable amount, and the grand total. Flag conflicts between English and Arabic fields, invalid dates, more than one payment method ticked, and TRNs that aren't 15 digits. Give each issue a severity."
   *Check:* I worked out line 3 and the VAT by hand. The AI first compared VAT only to the recomputed figure, so I added a check against the printed taxable amount to show the real cause: VAT was charged on the pre-discount total.

3. **Labelled data:** "Generate 45 invoice line descriptions for a UAE building-materials supplier, half English and half Arabic, using the 5 categories in this guide: <guide>. Include tricky ones such as delivery of a material."
   *Check:* I reviewed every row. I relabelled "delivery of cement bags" from cement to services, and kept the membrane out because it fits no class.

4. **Model + MLflow:** "Train TF-IDF + logistic regression with two feature settings (word vs char n-grams), log params, metrics and the model to MLflow (sqlite), choose by 5-fold CV macro-F1, and save reports."

5. **API:** "FastAPI /predict returning the category, confidence, top 3 and a needs_review flag below 0.5."
