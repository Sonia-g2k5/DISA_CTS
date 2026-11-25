
import os
import pandas as pd
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

load_dotenv()

USE_OLLAMA = False

if USE_OLLAMA:
    from langchain_community.llms import Ollama

    llm = Ollama(model="llama3")
    print("Using Ollama - LLaMA 3")

else:
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=api_key,
        temperature=0
    )

    print("Using Groq - LLaMA 3")


df = pd.read_excel(
    "MITRA_Inventory_Dataset.xlsx",
    sheet_name="Purchase Dataset",
    header=1
)

df.columns = [
    "retailer_id", "retailer_name", "district", "shop_name",
    "product_id", "product_name",
    "last_purchase_date", "restock_date", "qty_purchased",
    "interval_days", "notification_date", "days_before_restock",
    "notification_message"
]

date_columns = [
    "last_purchase_date",
    "restock_date",
    "notification_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")

print(f"Dataset loaded: {len(df)} records")


@tool
def check_restock_due(retailer_id: str) -> str:
    """Check products due for restocking for a specific retailer ID."""

    retailer_rows = df[df["retailer_id"].astype(str) == retailer_id]

    if retailer_rows.empty:
        return f"No records found for retailer {retailer_id}"

    today = pd.Timestamp.today().normalize()
    due_items = []

    for _, row in retailer_rows.iterrows():

        if pd.isna(row["restock_date"]):
            continue

        days_left = (row["restock_date"].normalize() - today).days

        if 0 <= days_left <= 5:
            due_items.append(
                f"{row['product_name']} - "
                f"Restock date: {row['restock_date'].strftime('%d %b %Y')} "
                f"({days_left} days remaining)"
            )

    if not due_items:
        return f"No products due for restocking within 5 days for {retailer_id}"

    return (
        f"Products due for retailer {retailer_id}:\n"
        + "\n".join(due_items)
    )


@tool
def get_retailer_info(retailer_id: str) -> str:
    """Get retailer name, shop name and district using retailer ID."""

    rows = df[df["retailer_id"].astype(str) == retailer_id]

    if rows.empty:
        return f"No retailer found with ID {retailer_id}"

    row = rows.iloc[0]

    return (
        f"Retailer: {row['retailer_name']}\n"
        f"Shop: {row['shop_name']}\n"
        f"District: {row['district']}"
    )


@tool
def generate_reminder_message(retailer_id: str, product_name: str) -> str:
    """Generate a personalized restocking reminder for a retailer."""

    rows = df[
        (df["retailer_id"].astype(str) == retailer_id) &
        (df["product_name"].astype(str).str.lower() == product_name.lower())
    ]

    if rows.empty:
        return f"No data found for {retailer_id} and {product_name}"

    row = rows.iloc[0]

    retailer_name = str(row["retailer_name"]).split()[0]
    product = row["product_name"]
    restock_date = row["restock_date"].strftime("%d %b %Y")
    interval = row["interval_days"]

    message = (
        f"Hi {retailer_name}! {product} is one of your regularly "
        f"purchased products. You usually restock it every "
        f"{interval} days. Your next restocking date is around "
        f"{restock_date}. Consider placing your order soon to "
        f"keep your shelves stocked. "
        f"Reorder link: [ORDER LINK]"
    )

    return message


@tool
def send_whatsapp_notification(retailer_id: str, message: str) -> str:
    """Simulate sending a WhatsApp notification."""

    rows = df[df["retailer_id"].astype(str) == retailer_id]

    if rows.empty:
        return f"Retailer {retailer_id} not found"

    retailer_name = rows.iloc[0]["retailer_name"]

    print("\n" + "-" * 50)
    print("WHATSAPP NOTIFICATION - DEMO MODE")
    print(f"To: {retailer_name}")
    print(f"Retailer ID: {retailer_id}")
    print(f"Message: {message}")
    print("Status: Simulated")
    print("-" * 50 + "\n")

    return f"Demo notification prepared for {retailer_name}"


@tool
def get_all_due_retailers(days_threshold: str = "5") -> str:
    """Find all retailers whose products are due for restocking."""

    today = pd.Timestamp.today().normalize()
    threshold = int(days_threshold)

    due_list = []

    for _, row in df.iterrows():

        if pd.isna(row["restock_date"]):
            continue

        days_left = (row["restock_date"].normalize() - today).days

        if 0 <= days_left <= threshold:
            due_list.append(
                f"Retailer ID: {row['retailer_id']}, "
                f"Retailer: {row['retailer_name']}, "
                f"Shop: {row['shop_name']}, "
                f"Product: {row['product_name']}, "
                f"Due in: {days_left} days"
            )

    if not due_list:
        return "No retailers are due for restocking within this period."

    return "Retailers due for restocking:\n" + "\n".join(due_list)


@tool
def check_new_arrivals(category: str) -> str:
    """Check simulated new product arrivals in a category."""

    new_arrivals = {
        "Soap": [
            "Dove Beauty Bar - New Variant",
            "Lifebuoy Neem Soap"
        ],
        "Shampoo": [
            "Keratin Smooth Shampoo - New"
        ],
        "Face Wash": [
            "Brightening Face Wash - New"
        ],
        "Toothpaste": [
            "Charcoal Toothpaste - New Variant"
        ]
    }

    category_key = next(
        (
            key for key in new_arrivals
            if key.lower() in category.lower()
        ),
        None
    )

    if not category_key:
        return f"No new arrivals found in category: {category}"

    buyers = df[
        df["product_name"].astype(str).str.contains(
            category_key, case=False, na=False
        )
    ]["retailer_id"].unique()

    return (
        f"New arrivals in {category_key}:\n"
        f"{', '.join(new_arrivals[category_key])}\n"
        f"Retailers who purchase this category: "
        f"{', '.join(map(str, buyers[:5]))}"
    )


tools = [
    check_restock_due,
    get_retailer_info,
    generate_reminder_message,
    send_whatsapp_notification,
    get_all_due_retailers,
    check_new_arrivals
]

prompt = PromptTemplate.from_template("""
You are MITRA, an intelligent inventory management assistant.

Your responsibilities:
- Check retailer restocking requirements.
- Identify products that need replenishment.
- Generate personalized restocking messages.
- Prepare simulated WhatsApp notifications.
- Identify retailers interested in new arrivals.

Use a friendly and professional tone.
Do not use warning or urgent alert language.

Available tools:
{tools}

Tool names: {tool_names}

Use this format:

Question: The task
Thought: Reason about the next step
Action: Tool name
Action Input: Tool input
Observation: Tool result

Repeat if necessary.

Thought: I now have the final answer
Final Answer: Provide the result.

Question: {input}
Thought: {agent_scratchpad}
""")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True
)

print("\n" + "=" * 50)
print("MITRA INVENTORY MANAGEMENT AGENT")
print("=" * 50)

tasks = [
    "Find all retailers whose products are due for restocking within the next 5 days.",
    "Check for new arrivals in the Soap category and identify retailers who may be interested.",
    "Show me the restocking details for the first retailer in the inventory dataset."
]

for index, task in enumerate(tasks, start=1):

    print(f"\nTASK {index}: {task}")
    print("-" * 50)

    try:
        result = agent_executor.invoke({"input": task})
        print("\nAGENT RESULT:")
        print(result["output"])

    except Exception as error:
        print(f"Task failed: {error}")

print("\nMITRA Agent execution completed.")
