import pandas as pd
import warnings
from datetime import datetime, timedelta
from twilio.rest import Client
warnings.filterwarnings("ignore")

print("=" * 60)
print("  DISHA — Agentic Inventory Management Agent")
print("  Powered by Ollama LLaMA 3 (Offline)")
print("=" * 60)

# ================================================================
#  TWILIO CREDENTIALS — FILL THESE 3 VALUES
# ================================================================
TWILIO_SID   = ""
TWILIO_TOKEN = ""
TWILIO_FROM  = ""
MY_NUMBER    = ""  # example: 9876543210
# ================================================================

print("\n Loading dataset...")
try:
    df = pd.read_excel(
        "MITRA_Inventory_Dataset.xlsx",
        sheet_name="Purchase Dataset",
        header=1
    )
except FileNotFoundError:
    print("\n ERROR: MITRA_Inventory_Dataset.xlsx not found!")
    print(" Make sure the Excel file is in D:\\disha\\ folder")
    exit()

df.columns = [str(c).strip() for c in df.columns]
print(f" Columns found: {list(df.columns)}")

# Normalize column names
col_map = {}
for c in df.columns:
    cl = c.lower().replace(" ", "_")
    col_map[c] = cl
df.rename(columns=col_map, inplace=True)

print(f" Dataset loaded: {len(df)} records")

# ── Connect to Ollama ─────────────────────────────────────────────
print("\n Connecting to Ollama LLaMA 3...")
try:
    import ollama as ollama_client
    ollama_client.chat(
        model="llama3",
        messages=[{"role": "user", "content": "say ok"}]
    )
    print(" Ollama connected successfully!\n")
    USE_OLLAMA = True
except Exception as e:
    print(f" Ollama not available: {e}")
    print(" Running in demo mode...\n")
    USE_OLLAMA = False


# ================================================================
#  AGENT TOOLS
# ================================================================

def check_restock_due(retailer_id):
    today = datetime.today()
    try:
        rows = df[df.iloc[:, 0].astype(str).str.strip() == retailer_id.strip()]
    except:
        rows = pd.DataFrame()

    if rows.empty:
        return f"No records found for {retailer_id}", [], "Unknown"

    due_items = []
    for _, row in rows.iterrows():
        try:
            restock = pd.to_datetime(row.iloc[8])
            days_left = (restock - today).days
            if days_left <= 5:
                due_items.append({
                    "product"     : str(row.iloc[6]),
                    "restock_date": restock.strftime("%d %b %Y"),
                    "days_left"   : days_left,
                    "interval"    : int(row.iloc[10]) if pd.notna(row.iloc[10]) else 10
                })
        except:
            continue

    if not due_items:
        try:
            nearest = rows.iloc[0]
            restock  = pd.to_datetime(nearest.iloc[8])
            days_left = (restock - today).days
            due_items.append({
                "product"     : str(nearest.iloc[6]),
                "restock_date": restock.strftime("%d %b %Y"),
                "days_left"   : days_left,
                "interval"    : int(nearest.iloc[10]) if pd.notna(nearest.iloc[10]) else 10
            })
        except:
            pass

    name = str(rows.iloc[0].iloc[1]) if not rows.empty else retailer_id
    result = f"Retailer {retailer_id} ({name}) restock status:\n"
    for item in due_items:
        result += (
            f"  Product : {item['product']}\n"
            f"  Restock : {item['restock_date']} ({item['days_left']} days away)\n"
        )
    return result, due_items, name


def generate_message(retailer_id, product_name, retailer_name,
                     restock_date, interval):
    fname = retailer_name.split()[0]

    if USE_OLLAMA:
        prompt = f"""
You are DISHA, a friendly business advisor for retail stores in India.
Write a short WhatsApp reminder message for a retailer.

Retailer first name : {fname}
Product             : {product_name}
Restock date        : {restock_date}
Purchase interval   : every {interval} days

Rules:
- Sound like a trusted friend giving business advice
- Mention it is a fast-selling product
- Say it is a good time to restock
- Mention maintaining profit
- End with: Tap to reorder: [ORDER LINK]
- Keep it under 5 sentences
- Do NOT use words like ALERT, WARNING, URGENT

Write only the message, nothing else.
"""
        try:
            response = ollama_client.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
        except:
            pass

    return (
        f"Hi {fname}! {product_name} is one of your fast-selling products. "
        f"We think it is a good time to restock before your shelves run low "
        f"— your usual reorder date is around {restock_date}. "
        f"To maintain your profit and keep your customers happy, "
        f"consider placing your order now. Tap to reorder: [ORDER LINK]"
    )


def send_whatsapp(retailer_id, retailer_name, shop_name, district, message):
    to_number = f"whatsapp:+91{MY_NUMBER}"
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body  = message,
            from_ = TWILIO_FROM,
            to    = to_number
        )
        print(f"\n  {'━'*54}")
        print(f"  DISHA — WhatsApp Notification SENT!")
        print(f"  {'━'*54}")
        print(f"  Retailer : {retailer_name} ({shop_name}, {district})")
        print(f"  Sent to  : +91{MY_NUMBER}")
        print(f"  SID      : {msg.sid}")
        print(f"\n  Message  :")
        print(f"  {message[:120]}...")
        print(f"  {'━'*54}\n")
        return True
    except Exception as e:
        print(f"\n  WhatsApp send FAILED: {e}")
        print(f"  Message (demo):\n  {message}\n")
        return False


# ================================================================
#  AGENTIC LOOP
# ================================================================

def run_agent_task(task_name, retailer_ids):
    print(f"\n{'='*60}")
    print(f"  DISHA AGENT — {task_name}")
    print(f"{'='*60}")

    notifications_sent = 0

    for rid in retailer_ids:
        print(f"\n  [PERCEIVE] Checking restock status for {rid}...")
        result, due_items, r_name = check_restock_due(rid)
        print(f"  Observation: {result}")

        if not due_items:
            print(f"  [REASON] No action needed for {rid}")
            continue

        try:
            rows     = df[df.iloc[:, 0].astype(str).str.strip() == rid]
            shop     = str(rows.iloc[0].iloc[3])
            district = str(rows.iloc[0].iloc[2])
        except:
            shop, district = "Unknown Shop", "Unknown District"

        for item in due_items:
            print(f"\n  [REASON] {item['product']} needs restock "
                  f"in {item['days_left']} days — generating message...")
            message = generate_message(
                rid, item["product"], r_name,
                item["restock_date"], item["interval"]
            )
            print(f"  [ACT] Sending WhatsApp notification...")
            send_whatsapp(rid, r_name, shop, district, message)
            notifications_sent += 1

    print(f"\n  Task complete — {notifications_sent} notifications sent")
    return notifications_sent


# ================================================================
#  RUN DISHA AGENT
# ================================================================

print("\n" + "="*60)
print("  Starting DISHA Agentic AI Loop")
print("="*60)

try:
    all_retailers = df.iloc[:, 0].dropna().astype(str).str.strip().unique()[:5]
    run_agent_task("Daily Restock Notifications", all_retailers)
except Exception as e:
    print(f"\nError running agent: {e}")
    print("Running with sample retailer IDs...")
    run_agent_task("Daily Restock Notifications", ["R001", "R002", "R003"])

print("\n" + "="*60)
print("  DISHA Agentic AI — All Tasks Complete")
print("  Perceive → Reason → Act loop ran successfully")
print("="*60)
