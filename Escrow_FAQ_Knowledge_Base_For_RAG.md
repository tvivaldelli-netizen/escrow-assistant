# Escrow FAQ Knowledge Base - For RAG Prototype

> **Purpose:** Content formatted for the Escrow Hub RAG prototype
> **Source:** Message Center Analysis (80,939 customer messages)
> **FAQs:** 19 total (7 existing + 12 new)

---

## How to Use This File

1. In your CC session with the trip planner repo, first explore the codebase
2. Then paste the FAQ content below into the knowledge base
3. Use the system prompt for the assistant persona
4. Test with the sample questions

---

## Escrow FAQ Knowledge Base

Add these 19 FAQs to the knowledge base. Each FAQ should be a separate chunk for retrieval.

### FAQ 1: What is a shortage, and why might I have one?
**Category:** Shortage

Your escrow account exists to pay escrowed items (taxes, insurance, mortgage insurance, etc.) as they come due. Shortages are commonly caused by increases in taxes or insurance, changing your insurance carrier off-cycle, or underestimated taxes when you closed your loan. A regular analysis of the escrow account is required to ensure that enough funds are available to pay these expenses as they come due. A shortage exists if the analysis shows that the minimum required balance falls below the amount required to make the projected payments of escrowed expenses.

---

### FAQ 2: Am I able to pay my shortage in a lump sum?
**Category:** Shortage

Yes. You may pay your shortage in full by submitting a payment to us. Please note that you are not required to pay your shortage in a lump sum. If you choose to pay your shortage in a lump sum, your monthly mortgage payment may still increase due to changes in your taxes and/or insurance. Making a payment is easy on your mobile app and website, or you can contact one of our Customer Care representatives and make a payment right over the phone.

---

### FAQ 3: Can I pay some of my shortage?
**Category:** Shortage

Yes. However, if you do elect to pay some of your shortage, please note that your monthly payment will not be reduced unless the entire shortage is paid in full.

---

### FAQ 4: Can I prepay my escrow amount so my payment does not change?
**Category:** Shortage

No. Taxes and insurance costs routinely change. The escrow portion of your monthly payment is collected so disbursements can be made when they're due – changes to the escrowed expenses will result in a change in your mortgage payment. While making supplemental payments toward your escrow may reduce or eliminate an escrow shortage, the escrow portion of your monthly payment is calculated annually in accordance with applicable law.

---

### FAQ 5: Can I pay an escrow shortage if my loan is 30+ days delinquent?
**Category:** Shortage

Payments on loans that are 30 or more days past due will first be credited toward any past-due amount owed before any funds are applied toward an escrow shortage.

---

### FAQ 6: Do I need to do anything if I have a bill pay service through my bank?
**Category:** Payment

You will need to contact your bank and change your monthly bill pay amount to your new monthly mortgage payment amount, based on the effective date.

---

### FAQ 7: Do I need to do anything if I'm enrolled in automatic payments?
**Category:** Payment

No action is needed if you have a shortage and want us to spread that amount over the next 12 months. We'll automatically adjust your monthly payment amount for you. If you'd like to voluntarily pay your shortage in a lump sum, instead, see the payment options available to you.

---

### FAQ 8: I changed insurance companies – what do I need to do?
**Category:** Insurance

If you've switched homeowners insurance carriers, please upload your new policy declarations page through your online account or mobile app. Once we receive and process your new policy information, your escrow account will be updated. If your new premium is significantly different from your previous policy, this may trigger an escrow analysis to adjust your monthly payment.

---

### FAQ 9: I received a refund check from my old insurance company – how do I apply it to my escrow?
**Category:** Insurance

If you received a refund from a previous insurance carrier, you can apply those funds to your escrow account by mailing a check with your loan number in the memo line to our payment processing address, or by contacting Customer Care to make a payment over the phone. Please note "Apply to Escrow" on your payment to ensure it is credited correctly.

---

### FAQ 10: My insurance or taxes went down – when will my payment decrease?
**Category:** Payment

Changes to your taxes or insurance will be reflected in your next annual escrow analysis. If the change is significant (typically $500 or more annually), you may request an off-cycle analysis. Please note that even if one expense decreases, your overall payment may not decrease if other escrowed expenses have increased or if you have an existing shortage.

---

### FAQ 11: How do I request a new escrow analysis?
**Category:** Analysis

Escrow analyses are performed annually, typically on your loan anniversary date. If you've had a significant change to your taxes or insurance (such as switching insurance carriers or receiving a tax exemption), you may request an off-cycle analysis by contacting Customer Care or submitting a message through your online account. Please allow 7-10 business days for processing after all updated documentation has been received.

---

### FAQ 12: Where can I see my current escrow balance?
**Category:** Balance

Your current escrow balance is available in your online account and mobile app. Navigate to the Escrow or Account Details section to view your balance, recent disbursements, and projected payments. Your balance is also shown on your monthly mortgage statement.

---

### FAQ 13: How much of my monthly payment goes to taxes versus insurance?
**Category:** Balance

You can view a breakdown of your escrow payment on your monthly statement or in your online account under the Escrow section. This breakdown shows the portion allocated to property taxes, homeowners insurance, and any other escrowed items such as mortgage insurance or flood insurance.

---

### FAQ 14: If I pay my shortage in full, what will my new monthly payment be?
**Category:** Payment

Your new monthly payment amount is shown on your escrow analysis statement under the option to pay the shortage in full. If you pay the full shortage amount, your payment will reflect only the increased escrow deposit needed for the coming year – you will not have the additional monthly shortage spread amount added. Please note that your payment may still be higher than your previous payment if your taxes or insurance have increased.

---

### FAQ 15: When will PMI be removed from my loan?
**Category:** PMI

For most conventional loans, Private Mortgage Insurance (PMI) is automatically cancelled when your loan balance reaches 78% of the original home value, based on your scheduled payments. You may also request cancellation when your balance reaches 80% of the original value. Please note that FHA loans have different mortgage insurance rules – FHA Mortgage Insurance Premium (MIP) may be required for the life of the loan depending on your down payment and loan terms.

---

### FAQ 16: How do I request PMI removal?
**Category:** PMI

You may submit a written request to remove PMI once your loan balance reaches 80% of your home's original value. Your loan must be current, with a good payment history. In some cases, an appraisal may be required to confirm your home's value. Contact Customer Care or submit a request through your online account to begin the process.

---

### FAQ 17: When will I receive my escrow refund after paying off my loan?
**Category:** Refund

After your loan is paid off, any remaining escrow balance will be refunded to you within 30 business days. The refund check will be mailed to the address on file. If you've recently moved, please ensure your mailing address is updated with us before your loan payoff.

---

### FAQ 18: How do I check if my homeowners insurance was paid from escrow?
**Category:** Status

You can view insurance disbursements in your online account under the Escrow Activity or Payment History section. This shows the date, payee, and amount of each escrow disbursement. If your insurance renewal date has passed and you don't see a payment, please contact Customer Care so we can verify your policy status.

---

### FAQ 19: How do I check the status of my escrow surplus or refund check?
**Category:** Status

If you're expecting a surplus refund check, you can contact Customer Care to confirm when it was mailed and the mailing address used. Surplus checks are typically mailed within 30 days of your annual escrow analysis. If your check has not arrived within 30 days of the mailing date, please contact us to request a replacement.

---

## Suggested System Prompt

```
You are an Escrow Assistant. Your role is to answer questions about escrow accounts, shortages, surpluses, insurance, taxes, PMI, and payment changes.

Guidelines:
- Be helpful, clear, and concise
- Use the knowledge base to answer questions accurately
- If a question is outside your knowledge base, acknowledge the limitation and suggest contacting Customer Care
- Never make up information about specific account details, balances, or dates
- Be empathetic – escrow can be confusing for customers

Topics you can help with:
- Escrow shortages and how to pay them
- Insurance changes and refunds
- Payment changes after escrow analysis
- PMI removal requests
- Escrow balance and disbursement questions
- Refund and surplus check status
```

---

## Test Questions

After setting up, test with these:

1. "What is an escrow shortage?"
2. "I switched insurance companies, what do I do?"
3. "When will my PMI be removed?"
4. "How can I pay my shortage?"
5. "Where do I find my escrow balance?"

---

## Categories Summary

| Category | FAQ Count | Topics |
|----------|-----------|--------|
| Shortage | 5 | What it is, paying lump sum, partial payment, prepay, delinquent |
| Payment | 5 | Bill pay, autopay, payment decrease, new payment amount, breakdown |
| Insurance | 2 | Changed carriers, refund deposit |
| PMI | 2 | Auto-removal, how to request |
| Balance | 2 | Where to find, tax vs insurance split |
| Analysis | 1 | Request new analysis |
| Refund | 1 | After payoff |
| Status | 2 | Insurance payment status, surplus check status |
