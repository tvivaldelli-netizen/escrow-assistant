Optional Airtable Integration

- File: `optional/airtable/airtable_integration.py`
- Install: `pip install pyairtable`
- Env (add to `backend/.env`):
  - `AIRTABLE_API_KEY=...`
  - `AIRTABLE_BASE_ID=...`
  - `AIRTABLE_TABLE_NAME=escrow_assistant_traces` (optional)

Usage
- Import and use in a custom route or workflow to log requests/responses for manual labeling.
- Not required for core escrow assistant; safe to ignore in production.

