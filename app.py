# app.py - Hugging Face Ready with .env

import os
import json
import pandas as pd
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import datetime
from dotenv import load_dotenv

# -----------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------
load_dotenv()  # Load from .env file

# Get API key from environment (works locally and on Hugging Face)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "JobYaari Career Assistant 🤖"
DATA_PATH = "jobyaari_full_dataset.json"

# Initialize Gemini LLM only if API key is available
llm = None
if GEMINI_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1
        )
        print("✅ Gemini AI initialized successfully")
    except Exception as e:
        print(f"❌ Gemini initialization failed: {e}")
        llm = None
else:
    print("❌ GEMINI_API_KEY not found in environment variables")

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} jobs")
    return df

df = load_data()

# -----------------------------
# IMPROVED RESPONSE FORMATTER
# -----------------------------
def format_job_cards(results_df):
    """Convert dataframe to beautiful job cards with apply buttons"""
    if results_df.empty:
        return "<div style='text-align: center; color: #64748b; padding: 20px;'>No jobs found 😔</div>"
    
    cards_html = "<div style='display: flex; flex-direction: column; gap: 15px;'>"
    
    for _, job in results_df.iterrows():
        cards_html += f"""
        <div style="background: linear-gradient(135deg, #1e3a5f, #2d4a7c); border-radius: 15px; padding: 20px; border: 1px solid #2563eb; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <h3 style="margin: 0 0 8px 0; color: #fbbf24;">{job['Title']}</h3>
                    <p style="margin: 5px 0; font-size: 14px; color: #cbd5e1;">🏢 {job['Organization']}</p>
                    <p style="margin: 5px 0; font-size: 14px; color: #cbd5e1;">📍 {job['Location']}</p>
                    <p style="margin: 5px 0; font-size: 14px; color: #cbd5e1;">💰 {job['Salary']}</p>
                    <p style="margin: 5px 0; font-size: 14px; color: #cbd5e1;">🎓 {job['Qualification']}</p>
                    <p style="margin: 5px 0; font-size: 14px; color: #cbd5e1;">⏰ Experience: {job['Experience']}</p>
                    <p style="margin: 5px 0; font-size: 14px; color: #fca5a5;">📅 Last Date: {job['Last Date']}</p>
                </div>
                <button style="background: linear-gradient(45deg, #f97316, #fb923c); border: none; border-radius: 10px; color: white; padding: 10px 20px; font-weight: bold; cursor: pointer; white-space: nowrap;">
                    Apply Now 🚀
                </button>
            </div>
        </div>
        """
    
    cards_html += "</div>"
    return cards_html

def create_funny_response(user_query, results_count, results_df):
    """Create engaging, humorous responses based on query and results"""
    query_lower = user_query.lower()
    
    # Handle specific question types with fun responses
    if any(word in query_lower for word in ["hello", "hi", "hey", "namaste"]):
        return f"👋 Hey there! Welcome to JobYaari! I found {results_count} awesome jobs waiting for you! What department are you exploring today? 😄", results_df
    
    elif any(word in query_lower for word in ["faq", "frequently asked", "help"]):
        return "📖 Our FAQ section is at www.jobyaari.com/faq - but honestly, I'm way more fun to talk to! 😉 What can I help you find? Engineering gigs? Science opportunities? Tell me! 🎯", results_df
    
    elif any(word in query_lower for word in ["engineering", "engineer"]):
        if results_count > 0:
            return f"🔧 Engineering whiz, huh? Fantastic! I found {results_count} engineering positions that might make your resume do a happy dance! 💃 What specific field in engineering interests you?", results_df
        else:
            return "🔧 Engineering is awesome! While I don't have specific engineering roles right now, check back soon - new opportunities pop up faster than bugs in production code! 🐛😄", results_df
    
    elif any(word in query_lower for word in ["science", "scientist"]):
        if results_count > 0:
            return f"🔬 Science enthusiast! Excellent! I found {results_count} science positions that are more exciting than a chemical reaction! 🧪 What specific area of science interests you?", results_df
        else:
            return "🔬 Science is amazing! While I don't have specific science roles right now, new discoveries (and jobs!) happen every day! 🔍 Check back soon!", results_df
    
    elif any(word in query_lower for word in ["get job", "will i get", "find job"]):
        return f"🎯 Will you get a job? With that awesome attitude - ABSOLUTELY! 🚀 I found {results_count} opportunities for you. The right job is like WiFi - sometimes you just need to move around a bit to find the best connection! 📶 Keep applying! ", results_df
    
    elif any(word in query_lower for word in ["total", "count", "how many"]):
        return f"📊 Woah! We've got {results_count} amazing opportunities in our database! That's like a buffet of career options 🍽️! What flavor are you craving today? 😄", results_df
    
    elif any(word in query_lower for word in ["thank", "thanks"]):
        return "🤗 You're welcome! Remember, I'm here 24/7 to help you find your dream job! Now go apply to those positions before someone else snacks on your opportunity! 🍩🚀", results_df
    
    elif any(word in query_lower for word in ["bye", "goodbye", "see you"]):
        return "👋 Bye bye! Don't be a stranger! Come back anytime you need job hunting support. Remember: Your next job is probably refreshing its browser waiting for YOU! 💻😄", results_df
    
    elif "experience" in query_lower and results_count > 0:
        return f"⏰ Got it! Looking for specific experience levels! I found {results_count} jobs matching your experience criteria! 🎯 Want to filter by location or qualification too?", results_df
    
    elif "qualification" in query_lower and results_count > 0:
        return f"🎓 Education matters! I found {results_count} jobs with qualifications that might match your background! 📚 Should we look at specific categories?", results_df
    
    # Default responses based on result count
    if results_count == 0:
        return "🤔 Hmm, I couldn't find exact matches for that. But don't worry! Try asking about 'Engineering jobs', 'Jobs in Delhi', 'Fresher opportunities' or 'Science jobs with 1 year experience'! I'm pretty good at those! 😊", results_df
    elif results_count == 1:
        return f"🎯 Found one perfect opportunity for you! This might be 'The One'! 💖 What other departments are you curious about?", results_df
    elif results_count <= 5:
        return f"✨ Found {results_count} awesome matches! Your skills are in demand! 🎉 What type of role gets you most excited?", results_df
    else:
        return f"🎊 WOW! Found {results_count} amazing jobs for you! Someone's got options! 😎 What specific field should we focus on?", results_df

# -----------------------------
# ENHANCED SEARCH FUNCTIONS
# -----------------------------
def enhanced_simple_search(user_query):
    """Enhanced fallback search with experience and qualification filtering"""
    query_lower = user_query.lower()
    results = df.copy()
    
    # Category filtering
    if "engineering" in query_lower:
        results = results[results['Category'] == 'Engineering']
    elif "science" in query_lower:
        results = results[results['Category'] == 'Science']
    elif "commerce" in query_lower:
        results = results[results['Category'] == 'Commerce']
    elif "education" in query_lower:
        results = results[results['Category'] == 'Education']
    
    # Location filtering
    if "delhi" in query_lower:
        results = results[results['Location'].str.contains('Delhi', case=False, na=False)]
    elif "bangalore" in query_lower or "bengaluru" in query_lower:
        results = results[results['Location'].str.contains('Bangalore|Bengaluru', case=False, na=False)]
    elif "mumbai" in query_lower:
        results = results[results['Location'].str.contains('Mumbai', case=False, na=False)]
    elif "chennai" in query_lower:
        results = results[results['Location'].str.contains('Chennai', case=False, na=False)]
    elif "hyderabad" in query_lower:
        results = results[results['Location'].str.contains('Hyderabad', case=False, na=False)]
    elif "kolkata" in query_lower:
        results = results[results['Location'].str.contains('Kolkata', case=False, na=False)]
    elif "pune" in query_lower:
        results = results[results['Location'].str.contains('Pune', case=False, na=False)]
    
    # Experience filtering - handle specific experience requirements
    if "1 year" in query_lower or "one year" in query_lower:
        results = results[results['Experience'].str.contains('1 year|1 Year|one year|1yr', case=False, na=False)]
    elif "2 year" in query_lower or "two year" in query_lower:
        results = results[results['Experience'].str.contains('2 year|2 Year|two year|2yr', case=False, na=False)]
    elif "3 year" in query_lower or "three year" in query_lower:
        results = results[results['Experience'].str.contains('3 year|3 Year|three year|3yr', case=False, na=False)]
    elif "fresher" in query_lower or "freshers" in query_lower or "no experience" in query_lower:
        results = results[results['Experience'].str.contains('fresher|Fresher|no experience|0 year', case=False, na=False)]
    elif "experience" in query_lower:
        # If user asks about experience but doesn't specify, show jobs with experience requirements
        results = results[~results['Experience'].str.contains('fresher|Fresher', case=False, na=False)]
    
    # Qualification filtering
    if "qualification" in query_lower or "education" in query_lower:
        # If user asks about qualifications, show jobs with specific qualification requirements
        if "b.tech" in query_lower or "btech" in query_lower or "b.e" in query_lower:
            results = results[results['Qualification'].str.contains('B.Tech|B.E|Engineering|BE|BTech', case=False, na=False)]
        elif "b.sc" in query_lower or "bsc" in query_lower:
            results = results[results['Qualification'].str.contains('B.Sc|Science|BSC', case=False, na=False)]
        elif "b.com" in query_lower or "bcom" in query_lower:
            results = results[results['Qualification'].str.contains('B.Com|Commerce|BCOM', case=False, na=False)]
        elif "m.tech" in query_lower or "mtech" in query_lower or "m.e" in query_lower:
            results = results[results['Qualification'].str.contains('M.Tech|M.E|ME|MTech', case=False, na=False)]
        elif "m.sc" in query_lower or "msc" in query_lower:
            results = results[results['Qualification'].str.contains('M.Sc|MSC', case=False, na=False)]
        elif "mba" in query_lower:
            results = results[results['Qualification'].str.contains('MBA', case=False, na=False)]
        elif "phd" in query_lower:
            results = results[results['Qualification'].str.contains('PhD|Ph.D', case=False, na=False)]
    
    # Latest notifications/updates
    if "latest" in query_lower or "recent" in query_lower or "new" in query_lower or "notifications" in query_lower:
        # Show all jobs for now (you can add date-based sorting later)
        results = results.head(15)  # Show first 15 as latest
    
    # Total count queries
    if "total" in query_lower or "count" in query_lower or "how many" in query_lower:
        results = df
    
    # Use our improved response formatter
    return create_funny_response(user_query, len(results), results)

# -----------------------------
# NLP-POWERED SEARCH WITH FALLBACK
# -----------------------------
def smart_search_with_nlp(user_query):
    """Use LangChain + Gemini with fallback to enhanced simple search"""
    
    # If Gemini is not available, use enhanced simple search
    if llm is None:
        return enhanced_simple_search(user_query)
    
    try:
        # Convert all jobs to context for Gemini
        jobs_context = ""
        for i, job in df.iterrows():
            jobs_context += f"""
Job {i}:
- Title: {job['Title']}
- Organization: {job['Organization']}
- Category: {job['Category']}
- Location: {job['Location']}
- Salary: {job['Salary']}
- Experience: {job['Experience']}
- Qualification: {job['Qualification']}
- Last Date: {job['Last Date']}
---
"""
        
        # Create NLP prompt for LangChain
        prompt = f"""
        USER QUERY: "{user_query}"
        
        JOB DATABASE:
        {jobs_context}
        
        INSTRUCTIONS:
        1. Understand the user's intent using NLP
        2. Find the most relevant job indices (0-based) from the database
        3. Return a natural, helpful response with emojis and friendly tone
        4. Always include matching indices
        5. Pay special attention to experience requirements, qualifications, categories, and locations
        
        RESPONSE FORMAT:
        ANSWER: [Your natural language response with emojis here]
        INDICES: [comma-separated numbers or "all"]
        """
        
        # Use LangChain to process with Gemini
        response = llm([HumanMessage(content=prompt)])
        response_text = response.content
        
        print(f"🤖 NLP Analysis: {response_text}")
        
        # Parse the response
        if "ANSWER:" in response_text and "INDICES:" in response_text:
            answer = response_text.split("ANSWER:")[1].split("INDICES:")[0].strip()
            indices_text = response_text.split("INDICES:")[1].strip()
            
            if indices_text.lower() == "all":
                results = df
            else:
                # Extract indices from response
                indices = []
                for part in indices_text.split(','):
                    part = part.strip()
                    if part.isdigit() and int(part) < len(df):
                        indices.append(int(part))
                
                if indices:
                    results = df.iloc[indices]
                else:
                    results = pd.DataFrame()
        else:
            # Fallback if format is wrong
            answer = "I found some relevant jobs for you:"
            results = df.head(10)
        
        # Apply our improved formatting to the response
        final_response = create_funny_response(user_query, len(results), results)
        return final_response
        
    except Exception as e:
        print(f"❌ NLP Error: {e}")
        # Fallback to enhanced simple search
        return enhanced_simple_search(user_query)

# -----------------------------
# WHATSAPP-STYLE UI
# -----------------------------
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0a192f 0%, #172a45 100%) !important;
    font-family: 'Segoe UI', system-ui !important;
}

.contain {
    background: #0a192f !important;
    border-radius: 15px !important;
}

.message-bot {
    background: #1e3a5f !important;
    color: white !important;
    border-radius: 18px 18px 18px 4px !important;
    margin: 12px 0 !important;
    padding: 14px 18px !important;
    max-width: 80% !important;
    border: 1px solid #2563eb !important;
    margin-right: auto !important;
}

.message-user {
    background: #f97316 !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    margin: 12px 0 !important;
    padding: 14px 18px !important;
    max-width: 80% !important;
    margin-left: auto !important;
}

.message-time {
    font-size: 0.75em !important;
    color: #94a3b8 !important;
    margin-top: 6px !important;
}

.input-box {
    background: #1e293b !important;
    border: 2px solid #334155 !important;
    border-radius: 25px !important;
    color: white !important;
    padding: 16px 20px !important;
}

.button-primary {
    background: linear-gradient(45deg, #f97316, #fb923c) !important;
    border: none !important;
    border-radius: 25px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 16px 32px !important;
}

.dataframe {
    background: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
}

.dataset-item {
    background: #1e3a5f !important;
    color: white !important;
    border: 1px solid #2563eb !important;
    border-radius: 20px !important;
    margin: 6px !important;
    padding: 12px 18px !important;
    cursor: pointer !important;
}

.job-cards-container {
    background: #0a192f !important;
    padding: 20px !important;
    border-radius: 15px !important;
    margin-top: 20px !important;
}
"""

def create_chat_ui():
    with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
        
        ai_status = "🧠 Powered by Google Gemini AI" if llm else "🔧 Basic Search Mode"
        
        gr.Markdown(f"""
        <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #0a192f, #1e3a5f); border-radius: 16px; margin-bottom: 25px; border: 1px solid #2563eb;">
            <h1 style="color: white; margin: 0 0 10px 0; font-size: 32px;">{APP_TITLE}</h1>
            <p style="color: #f97316; margin: 5px 0; font-size: 18px;">{ai_status}</p>
            <p style="color: #94a3b8; font-size: 14px;">{len(df)} jobs available • Your career buddy with jokes! 😄</p>
        </div>
        """)
        
        chat_state = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                <div style="background: #1e293b; padding: 20px; border-radius: 15px; border: 1px solid #334155;">
                    <h3 style="color: #f97316; margin-top: 0;">🚀 Quick Search</h3>
                    <p style="color: #94a3b8; font-size: 14px;">Click any option:</p>
                </div>
                """)
                
                suggestions = gr.Dataset(
                    components=[gr.Textbox(visible=False)],
                    samples=[
                        ["Show all jobs"],
                        ["Engineering jobs"],
                        ["Science jobs"], 
                        ["Commerce jobs"],
                        ["Education jobs"],
                        ["Jobs in Delhi"],
                        ["Fresher jobs"],
                        ["Science jobs with 1 year experience"],
                        ["Latest engineering notifications"],
                        ["Jobs requiring B.Tech qualification"]
                    ],
                    label=""
                )
            
            with gr.Column(scale=2):
                chat_display = gr.HTML(
                    value="<div style='text-align: center; color: #64748b; padding: 40px; font-size: 16px;'>💬 Hey there! I'm your JobYaari buddy! Ask me about jobs and I'll try to make you smile too! 😊</div>"
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask about jobs... (I understand natural language and jokes!)",
                        lines=2,
                        container=False,
                        elem_classes="input-box",
                        scale=4
                    )
                    send_btn = gr.Button("Send 🚀", elem_classes="button-primary", scale=1)
                
                with gr.Accordion("📊 Job Results", open=True):
                    results_display = gr.HTML()
        
        def process_message(user_message, history):
            if not user_message.strip():
                return history, "", history, ""
            
            current_time = datetime.datetime.now().strftime("%H:%M")
            
            # Get search results
            bot_response, results_df = smart_search_with_nlp(user_message)
            
            # Format job cards
            job_cards = format_job_cards(results_df)
            
            # Add to history
            new_entry = {
                "user": user_message,
                "bot": bot_response, 
                "time": current_time,
                "results_count": len(results_df)
            }
            history.append(new_entry)
            
            # Generate chat HTML
            chat_html = "<div style='padding: 20px; min-height: 400px;'>"
            for msg in history:
                chat_html += f"""
                <div style='display: flex; justify-content: flex-end; margin: 20px 0;'>
                    <div class='message-user'>
                        <div style='font-size: 14px;'>{msg['user']}</div>
                        <div class='message-time'>{msg['time']}</div>
                    </div>
                </div>
                """
                chat_html += f"""
                <div style='display: flex; justify-content: flex-start; margin: 20px 0;'>
                    <div class='message-bot'>
                        <div style='font-size: 14px;'>{msg['bot']}</div>
                        <div class='message-time'>{msg['time']} • {msg['results_count']} jobs found</div>
                    </div>
                </div>
                """
            chat_html += "</div>"
            
            return history, "", chat_html, job_cards
        
        def on_suggestion_click(evt: gr.SelectData, history):
            return evt.value[0], history
        
        send_btn.click(process_message, [chat_input, chat_state], [chat_state, chat_input, chat_display, results_display])
        chat_input.submit(process_message, [chat_input, chat_state], [chat_state, chat_input, chat_display, results_display])
        suggestions.select(on_suggestion_click, [chat_state], [chat_input, chat_state])
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting JobYaari Chatbot...")
    print(f"📊 Loaded {len(df)} jobs")
    if llm:
        print("🧠 Gemini AI: Active")
    else:
        print("🔧 Basic Search: Active (Gemini API key not found)")
    print("🌐 Server: http://127.0.0.1:7860")
    
    demo = create_chat_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)