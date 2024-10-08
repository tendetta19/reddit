import os
import re
import io
import time
import traceback
from datetime import datetime
from functools import wraps
from urllib.parse import urlparse 
import streamlit as st
import praw
from openai import AzureOpenAI
from dotenv import load_dotenv
from openpyxl import load_workbook
from stqdm import stqdm  # For progress bar in Streamlit
import plotly.express as px  # For pie chart visualization
import pandas as pd
import concurrent.futures  # For parallel processing
import plotly.io as pio


# ==============================================================================
# Configuration and Initialization
# ==============================================================================

# Load environment variables from .env file
load_dotenv()

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT')
)

# Azure OpenAI configuration
deployment_name = "gpt-4"
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_api_key,
    api_version="2024-05-01-preview",
) 

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_submission_id(url):
    """Extract Reddit submission ID from URL."""
    try:
        parsed_url = urlparse(url)
        match = re.search(r'/comments/([a-zA-Z0-9]+)', parsed_url.path)
        return match.group(1) if match else None
    except Exception as e:
        st.error(f"Error parsing URL: {e}")
        return None

def fetch_reddit_post(submission_id):
    """Fetch Reddit post details using submission ID."""
    try:
        submission = reddit.submission(id=submission_id)
        return (submission.title, submission.selftext, submission.author.name) if submission else (None, None, None)
    except Exception as e:
        st.error(f"Error fetching Reddit post: {e}")
        return None, None, None

def fetch_all_comments(submission_id, limit=50):
    """
    Fetch a limited number of comments from a Reddit submission, excluding deleted comments.

    Args:
        submission_id (str): The Reddit submission ID.
        limit (int): The maximum number of comments to retrieve.

    Returns:
        list: A list of comment bodies excluding deleted comments.
    """
    try:
        submission = reddit.submission(id=submission_id)
        submission.comments.replace_more(limit=0)  # Avoid fetching "MoreComments" objects
        comments = []
        for comment in submission.comments.list():
            if comment.body in ("[deleted]", "[removed]"):
                continue

            comments.append(comment.body)
            if len(comments) >= limit:
                break
        return comments
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []


def write_report_to_excel(report, template_path):
    """Write report data to Excel using a template."""
    try:
        workbook = load_workbook(template_path)
        # Access the 'Stats' sheet
        stats_sheet = workbook['Stats']
        main_sheet = workbook.active  # Assuming the main sheet is the active one

        # Populate the required fields in the main sheet
        main_sheet['B2'] = report['Title']
        main_sheet['B3'] = report['Link']
        main_sheet['B4'] = report['Body'] 
        main_sheet['B9'] = report['Content Type Score']
        main_sheet['D9'] = report['Content Type Explanation']
        main_sheet['B10'] = report['Sentiment Score']
        main_sheet['D10'] = report['Sentiment Explanation']
        
        main_sheet['B11'] = report['Targeting Score']
        main_sheet['D11'] = report['Targeting Explanation']
        main_sheet['B12'] = report['Context Score']
        main_sheet['D12'] = report['Context Explanation']
        main_sheet['B13'] = report['Urgency Score']
        main_sheet['D13'] = report['Urgency Explanation']
        main_sheet['B16'] = report["Summary"]

       
        # Write the counts into the 'Stats' sheet
        stats_sheet['C19'] = report.get('Agree Count', 0)
        stats_sheet['C20'] = report.get('Disagree Count', 0)
        stats_sheet['C21'] = report.get('Neutral Count', 0)

        # Write the general consensus into cell E19
        stats_sheet['E19'] = report.get('General Consensus', 'N/A')

        # Save the workbook to a BytesIO buffer
        output = io.BytesIO()
        workbook.save(output)
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error while generating the Excel report: {e}")
        traceback.print_exc()
        return None

# ==============================================================================
# Analysis Functions
# ==============================================================================

def analyze_content_type(title, body):
    """Analyze content type using Azure OpenAI."""
    prompt = f"""
    Based on the following Reddit post, analyze its content type and classify it on a scale of 1 to 10:

    1: No violation.
    2: Slightly controversial.
    3: Minor sensitivity.
    4: Somewhat controversial.
    5: Sensitive topics or controversial subjects.
    6: Subtle references to controversial topics.
    7: Sensitive or slightly problematic content.
    8: References problematic content without being harmful.
    9: Strong hints at violations (e.g., borderline threats).
    10: Extreme violations (e.g., threats, slurs).

    Title: {title}

    Body: {body}

    Return a single number from 1 to 10 as the classification, followed by an explanation in the format: 1| Explanation
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        rating, explanation = response.choices[0].message.content.strip().split('|', 1)
        return int(rating.strip()), explanation.strip()
    except ValueError:
        st.error("Error parsing content type analysis response.")
        return 0, "No explanation available."

def analyze_sentiment(body):
    """Analyze sentiment using Azure OpenAI."""
    prompt = f"""
    Based on the following Reddit post, analyze its sentiment and classify it on a scale of 1 to 10:

    1: Very positive.
    2: Positive with minor criticism.
    3: Mostly positive with some negativity.
    4: Neutral with slight positive undertones.
    5: Neutral.
    6: Slightly negative.
    7: Mostly negative with some positive aspects.
    8: Predominantly negative.
    9: Strongly negative.
    10: Extremely negative or hostile tone.

    Body: {body}

    Return a single number from 1 to 10 as the classification, followed by an explanation in the format: 1| Explanation
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        rating, explanation = response.choices[0].message.content.strip().split('|', 1)
        return int(rating.strip()), explanation.strip()
    except ValueError:
        st.error("Error parsing sentiment analysis response.")
        return 0, "No explanation available."


def analyze_targeting(title, body):
    """Analyze targeting using Azure OpenAI."""
    prompt = f"""
    Based on the following Reddit post, analyze if there is any targeting of specific individuals or groups.

    1: No targeting detected.
    2: Very subtle targeting.
    3: Occasional indirect targeting.
    4: Frequent indirect references.
    5: Subtle targeting.
    6: Direct but mild targeting.
    7: Direct targeting without aggressive language.
    8: Strong targeting.
    9: Aggressive targeting, borderline harmful.
    10: Explicit targeting with harmful intent.

    Title: {title}
    Body: {body}

    Return a single number from 1 to 10 as the classification, followed by an explanation in the format: 1| Explanation
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        rating, explanation = response.choices[0].message.content.strip().split('|', 1)
        return int(rating.strip()), explanation.strip()
    except ValueError:
        st.error("Error parsing targeting analysis response.")
        return 0, "No explanation available."

def analyze_context(title, body):
    """Determine context relevance using Azure OpenAI."""
    prompt = f"""
    Based on the following Reddit post, analyze its relevance to ongoing societal or political issues.

    1: No relevance.
    2: Minor relevance.
    3: Occasionally relevant to sensitive topics.
    4: Moderately relevant.
    5: Current events discussed but not central.
    6: Important events mentioned but not central.
    7: Highly relevant to current events.
    8: Significant relevance to major issues.
    9: Strongly tied to current issues.
    10: Direct relevance to critical issues.

    Title: {title}
    Body: {body}

    Return a single number from 1 to 10 as the classification, followed by an explanation in the format: 1| Explanation
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        rating, explanation = response.choices[0].message.content.strip().split('|', 1)
        return int(rating.strip()), explanation.strip()
    except ValueError:
        st.error("Error parsing context analysis response.")
        return 0, "No explanation available."

def analyze_urgency(title, body):
    """Assess urgency of threat using Azure OpenAI."""
    prompt = f"""
    Based on the following Reddit post, analyze if there is any sense of urgency or immediate threat.

    1: No threat detected.
    2: Minor hypothetical mentions.
    3: Occasional mention of potential threat.
    4: Slight undertones of urgency.
    5: Hypothetical threat.
    6: Subtle mentions of urgency.
    7: Strong hints of an issue.
    8: High urgency but not immediate.
    9: Immediate threat implied.
    10: Immediate threat, calls for action.

    Title: {title}
    Body: {body}

    Return a single number from 1 to 10 as the classification, followed by an explanation in the format: 1| Explanation
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        rating, explanation = response.choices[0].message.content.strip().split('|', 1)
        return int(rating.strip()), explanation.strip()
    except ValueError:
        st.error("Error parsing urgency analysis response.")
        return 0, "No explanation available."



def extract_problem_statement(title, body):
    """
    Extract the problem statement from the Reddit post body using Azure OpenAI.

    Args:
        title (str): The title of the Reddit post.
        body (str): The body of the Reddit post.

    Returns:
        str: The extracted problem statement.
    """
    prompt = f"""
    From the following Reddit post, extract the main problem statement.

    Title: {title}

    Body:
    {body}

    Please provide only the problem statement in one clear and concise sentence.
    """
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}]
        )
        problem_statement = response.choices[0].message.content.strip()
        return problem_statement
    except Exception as e:
        st.error(f"Error extracting problem statement: {e}")
        return "Problem statement could not be extracted."

def analyze_comment_stance(comment_body, problem_statement, post_body):
    """
    Analyze the stance of a comment relative to the problem statement and post body using individual scores for each criterion.

    Args:
        comment_body (str): The text of the comment.
        problem_statement (str): The extracted problem statement from the post.
        post_body (str): The body text of the original post.

    Returns:
        str: The stance of the comment ('Agree', 'Disagree', 'Neutral').
    """
    prompt = f"""
    You are an assistant tasked with scoring a Reddit comment in relation to the original post.

    **Problem Statement:**
    {problem_statement}

    **Original Post Body:**
    {post_body}

    **Comment:**
    {comment_body}

    **Instructions:**
    Analyze the comment based on the following criteria, and assign a score between -10 and 10 for each:

    1. **Similarity to the contents of the post:**
       - Evaluate how closely the comment relates to the topics and points discussed in the post.
       - **Score:** -10 (not similar at all) to 10 (highly similar).

    2. **Tone towards the post:**
       - Determine if the comment expresses a positive, negative, or neutral sentiment towards the post.
       - **Score:** -10 (very negative) to 10 (very positive).

    3. **Additional Factors:**
       - Consider any other elements that might indicate the comment's stance (e.g., supportive language, criticism, questions).
       - **Score:** -10 (very unfavorable) to 10 (very favorable).

    **Your Task:**
    Provide the score for each criterion in the following format:

    **Scores:**
    Similarity Score: [score]
    Tone Score: [score]
    Additional Factors Score: [score]

    **Total Score:**
    [sum of the three scores]

    **Instructions:**
    - Make sure each score is an integer between -10 and 10.
    - Sum the individual scores to get the total score.

    **Response Format:**
    Provide only the scores as specified without additional commentary.
    """

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()

        # Extract the individual scores 
        similarity_score_match = re.search(r"Similarity Score:\s*(-?\d+)", content)
        tone_score_match = re.search(r"Tone Score:\s*(-?\d+)", content)
        additional_score_match = re.search(r"Additional Factors Score:\s*(-?\d+)", content)

        if similarity_score_match and tone_score_match and additional_score_match:
            similarity_score = int(similarity_score_match.group(1))
            tone_score = int(tone_score_match.group(1))
            additional_score = int(additional_score_match.group(1))

            total_score = similarity_score + tone_score + additional_score

            # Ensure total score is between -30 and 30
            total_score = max(min(total_score, 30), -30)

            if total_score > 0:
                return 'Agree'
            elif total_score < 0:
                return 'Disagree'
            else:
                return 'Neutral'
        else:
            # Handle unexpected responses by defaulting to 'Neutral'
            return 'Neutral'
    except Exception as e:
        st.error(f"Error analyzing comment stance: {e}")
        return 'Neutral'
def calculate_risk_level(report):
    """Calculate the risk level based on total score."""
    total_score = sum([
        report['Content Type Score'],
        report['Sentiment Score'],
        report['Targeting Score'],
        report['Context Score'],
        report['Urgency Score']
    ])
    percentage = (total_score / 70) * 100  # Since maximum total score is 70
    if percentage > 66:
        risk_level = 'HIGH'
    elif 33 < percentage <= 66:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    return risk_level, percentage

def generate_summary(report, risk_level):
    """Generate summary using Azure OpenAI based on the analysis results."""
    prompt = f"""
    Based on the following analysis of a Reddit post, write a concise summary paragraph:

    Title: {report['Title']}
    Problem Statement: {report.get('Problem Statement', 'N/A')}
    Content Type Score: {report['Content Type Score']} - {report['Content Type Explanation']}
    Sentiment Score: {report['Sentiment Score']} - {report['Sentiment Explanation']}
    
    Targeting Score: {report['Targeting Score']} - {report['Targeting Explanation']}
    Context Score: {report['Context Score']} - {report['Context Explanation']}
    Urgency Score: {report['Urgency Score']} - {report['Urgency Explanation']}
    Agree Percentage: {report.get("Agree Percentage", 0):.2f}% 
    Disagree Percentage: {report.get("Disagree Percentage", 0):.2f}%
    Neutral Percentage: {report.get("Neutral Percentage", 0):.2f}%

    Write a summary paragraph that provides an overview of the Reddit post, highlighting key findings from the analysis. The summary should be clear, concise, and written in third person. Also include any action items if the post is alarming, such as whether third-party intervention is required.
    """
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Summary could not be generated."

def analyze_general_consensus(comments):
    """
    Determine the general consensus of the comments using OpenAI by analyzing all comment texts.

    Args:
        comments (list): List of comment texts.

    Returns:
        str: A summary statement of the general consensus.
    """
    # Due to potential token limits, we'll summarize comments in batches if necessary
    MAX_TOKENS = 3000  # Adjust based on model's token limit
    BATCH_SIZE = 100    # Number of comments per batch; adjust as needed

    def summarize_comments(comment_batch):
        """Summarize a batch of comments."""
        prompt = f"""
        You are an analyst tasked with summarizing the general consensus of comments on a Reddit post.

        Here are some comments:
        {comment_batch}

        Based on these comments, provide a concise summary statement that reflects how the general public is feeling about the post. The summary should be clear, neutral, and informative on things they are agreeing on, disagreeing on, or are neutral about.
        """

        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,  # Adjust as needed
                temperature=0.5
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            st.error(f"Error summarizing comments: {e}")
            return ""

    # If the number of comments is large, process them in batches
    summaries = []
    for i in range(0, len(comments), BATCH_SIZE):
        batch = comments[i:i+BATCH_SIZE]
        # Concatenate comments separated by newlines
        comment_text = "\n".join(batch)
        summary = summarize_comments(comment_text)
        if summary:
            summaries.append(summary)

    # Now, summarize all batch summaries into a final consensus
    aggregated_summary = "\n".join(summaries)

    final_prompt = f"""
    You have the following summaries of Reddit comments:

    {aggregated_summary}

    Based on these summaries, provide a final concise summary statement that reflects the overall general consensus of all comments on the Reddit post. The summary should be clear, neutral, and informative about the main points of agreement, disagreement, or neutrality among the commenters.
    """

    try:
        final_response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=300,  # Adjust as needed
            temperature=0.5
        )
        final_consensus = final_response.choices[0].message.content.strip()
        return final_consensus
    except Exception as e:
        st.error(f"Error generating final consensus: {e}")
        return "General consensus could not be determined."

# ==============================================================================
# Retry Decorator
# ==============================================================================

def retry_on_exception(retries=3, delay=1):
    """Decorator to retry a function on exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with st.spinner("Generation of report in progress. Please wait..."):
                for attempt in range(1, retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        st.warning(f"An error occurred: {e}. Retrying ({attempt}/{retries})...")
                        time.sleep(delay)
                        if attempt == retries:
                            st.error("Failed after multiple attempts.")
                            raise
        return wrapper
    return decorator

# ==============================================================================
# Report Generation
# ==============================================================================

@st.cache_data(show_spinner=False)
@retry_on_exception(retries=3, delay=1)
def generate_report(title, body, author, reddit_url): 
    
    # Existing analyses
    content_type_score, content_type_explanation = analyze_content_type(title, body)
    sentiment_score, sentiment_explanation = analyze_sentiment(body)
    targeting_score, targeting_explanation = analyze_targeting(title, body)
    context_score, context_explanation = analyze_context(title, body)
    urgency_score, urgency_explanation = analyze_urgency(title, body) 

    # Extract problem statement
    problem_statement = extract_problem_statement(title, body)

    # Fetch and analyze comments
    submission_id = get_submission_id(reddit_url)
    comments = fetch_all_comments(submission_id)
    comment_analysis = []
    agree_count = 0
    disagree_count = 0
    neutral_count = 0

 
    if comments:
        st.info("Analyzing comments' stances. This may take a while depending on the number of comments...")
 

        total_comments = len(comments)
        max_workers = min(32, os.cpu_count() + 4)  # Adjust based on your system and API rate limits

        # Initialize Streamlit progress bar and placeholder
        progress_bar = st.progress(0)
        remaining_placeholder = st.empty()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_comment = {
                executor.submit(analyze_comment_stance, comment, problem_statement, body): comment
                for comment in comments
            }

            completed = 0
            for future in concurrent.futures.as_completed(future_to_comment):
                comment = future_to_comment[future]  # Retrieve the original comment 
                try:
                    stance = future.result()  
                    comment_analysis.append({'comment': comment, 'stance': stance})

 
                    if stance == 'Agree':
                        agree_count += 1
                    elif stance == 'Disagree':
                        disagree_count += 1
                    else:
                        neutral_count += 1

                except Exception as e:
                    st.error(f"Error processing comment: {e}")

                # Update progress
                completed += 1
                progress_percentage = completed / total_comments
                progress_bar.progress(progress_percentage)

                # Update remaining comments
                remaining = total_comments - completed
                remaining_placeholder.text(f"Comments left to process: {remaining}")

        # Final update after processing is complete
        remaining_placeholder.text("All comments have been processed.")
        progress_bar.empty()  # Optionally remove the progress bar

        # Calculate percentages
        agree_percentage = (agree_count / total_comments) * 100 if total_comments else 0
        disagree_percentage = (disagree_count / total_comments) * 100 if total_comments else 0
        neutral_percentage = (neutral_count / total_comments) * 100 if total_comments else 0

    else:
        agree_percentage = disagree_percentage = neutral_percentage = 0


    # Determine General Consensus using OpenAI based on all comments
    general_consensus = analyze_general_consensus(comments)

    report = {
        "Title": title,
        "Link": reddit_url,
        "Body": body,
        "Problem Statement": problem_statement,  # New field
        "Content Type Score": content_type_score,
        "Content Type Explanation": content_type_explanation,
        "Sentiment Score": sentiment_score,
        "Sentiment Explanation": sentiment_explanation,
        "Targeting Score": targeting_score,
        "Targeting Explanation": targeting_explanation,
        "Context Score": context_score,
        "Context Explanation": context_explanation,
        "Urgency Score": urgency_score,
        "Urgency Explanation": urgency_explanation,
        "Comments": comment_analysis,
        "Agree Percentage": agree_percentage,
        "Disagree Percentage": disagree_percentage,
        "Neutral Percentage": neutral_percentage,
        "Agree Count": agree_count,
        "Disagree Count": disagree_count,
        "Neutral Count": neutral_count,
        "General Consensus": general_consensus  # Added field
    }

    # Calculate risk level
    risk_level, percentage = calculate_risk_level(report)
    report["Risk Level"] = risk_level
    report["Risk Percentage"] = percentage

    # Generate summary with risk level
    report["Summary"] = generate_summary(report, risk_level)
    return report

# ==============================================================================
# Streamlit UI Components
# ==============================================================================
def render_sidebar():
    """Render the scoring metric explanations in the sidebar as a single collapsible section with basic explanations."""
    st.sidebar.title("Scoring Metrics")

    with st.sidebar.expander("View Scoring Metric Explanations", expanded=False):
        # Content Type
        st.markdown("### Content Type (1–10)")
        st.markdown("""
        ***Content Type** assesses the nature and severity of the content in the post, categorizing it based on the presence of violations, controversial topics, or harmful language.*

        - **1:** No violation.
        - **2:** Slightly controversial.
        - **3:** Minor sensitivity.
        - **4:** Somewhat controversial.
        - **5:** Sensitive topics or controversial subjects.
        - **6:** Subtle references to controversial topics.
        - **7:** Sensitive or slightly problematic content.
        - **8:** References problematic content without being harmful.
        - **9:** Strong hints at violations (e.g., borderline threats).
        - **10:** Extreme violations (e.g., threats, slurs).
        """)

        # Sentiment
        st.markdown("### Sentiment (1–10)")
        st.markdown("""
        ***Sentiment** evaluates the overall emotional tone of the post, determining whether it conveys positive, negative, or neutral sentiments.*

        - **1:** Very positive.
        - **2:** Positive with minor criticism.
        - **3:** Mostly positive with some negativity.
        - **4:** Neutral with slight positive undertones.
        - **5:** Neutral.
        - **6:** Slightly negative.
        - **7:** Mostly negative with some positive aspects.
        - **8:** Predominantly negative.
        - **9:** Strongly negative.
        - **10:** Extremely negative or hostile tone.
        """)
 
        # Targeting
        st.markdown("### Targeting (1–10)")
        st.markdown("""
        ***Targeting** assesses whether the post directs negative attention toward specific individuals or groups, evaluating the level of aggression or hostility.*

        - **1:** No targeting.
        - **2:** Very subtle targeting.
        - **3:** Occasional indirect targeting.
        - **4:** Frequent indirect references.
        - **5:** Subtle targeting.
        - **6:** Direct but mild targeting.
        - **7:** Direct targeting without aggressive language.
        - **8:** Strong targeting.
        - **9:** Aggressive targeting, borderline harmful.
        - **10:** Explicit targeting with harmful intent.
        """)

        # Context
        st.markdown("### Context (1–10)")
        st.markdown("""
        ***Context** evaluates the relevance of the post to current societal or political events, determining how closely it aligns with ongoing discussions or issues.*

        - **1:** No relevance to societal events.
        - **2:** Minor relevance.
        - **3:** Occasionally relevant to sensitive topics.
        - **4:** Moderately relevant.
        - **5:** Current events discussed but not central.
        - **6:** Important events mentioned but not central.
        - **7:** Highly relevant to current events.
        - **8:** Significant relevance to major issues.
        - **9:** Strongly tied to current issues.
        - **10:** Direct relevance to critical issues.
        """)

        # Urgency
        st.markdown("### Urgency (1–10)")
        st.markdown("""
        ***Urgency** gauges the immediacy of a threat or the need for prompt action within the post, indicating the level of concern it may raise.*

        - **1:** No urgency.
        - **2:** Minor hypothetical mentions.
        - **3:** Occasional mention of potential threat.
        - **4:** Slight undertones of urgency.
        - **5:** Hypothetical threat.
        - **6:** Subtle mentions of urgency.
        - **7:** Strong hints of an issue.
        - **8:** High urgency but not immediate.
        - **9:** Immediate threat implied.
        - **10:** Immediate threat, calls for action.
        """)

# ==============================================================================
# Main Application
# ==============================================================================\


# Set the page configuration
st.set_page_config(
    page_title="Reddit Incident Analyzer",  # The title displayed on the browser tab
    page_icon="🔍",                          # Optional: Add an emoji or image as the tab icon
    layout="wide",                           # Optional: Choose between 'centered' or 'wide' layout
    initial_sidebar_state="expanded"         # Optional: Set the initial state of the sidebar
)

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state variables
    if 'previous_link' not in st.session_state:
        st.session_state['previous_link'] = ''
    if 'report' not in st.session_state:
        st.session_state['report'] = None

    # Create a navigation menu in the sidebar
    st.sidebar.title("Navigation")
    nav_options = ["Generate Report", "Full Report", "Recommendations", "Detailed Analysis", "Problem Statement", "Comment Stance Analysis"]
    selection = st.sidebar.radio("Go to", nav_options)

    # Render the sidebar
    render_sidebar()
    
    # Title with optional image or logo
    st.markdown("<h1 style='text-align: center;'>🔍 Reddit Incident Analysis Report</h1>", unsafe_allow_html=True)
    st.write("")  # Add a spacer

    if selection == "Generate Report":
        # Input for Reddit URL
        st.header("Generate Report")
        st.write("Enter the Reddit post URL below to generate an incident analysis report.")
        reddit_url = st.text_input("Enter a Reddit post URL:")

        # Add help expander
        with st.expander("Need help? Click here for sample Reddit links"):
            st.markdown("Here are some sample Reddit links you can try:")
            st.write("[https://www.reddit.com/r/changemyview/comments/xnwpqu/cmv_the_uk_royal_family_are_an_outdated_and/](https://www.reddit.com/r/changemyview/comments/xnwpqu/cmv_the_uk_royal_family_are_an_outdated_and/)")
            st.write("[https://www.reddit.com/r/malaysia/comments/1fsu4aq/does_chinese_employers_really_have_vendetta/](https://www.reddit.com/r/malaysia/comments/1fsu4aq/does_chinese_employers_really_have_vendetta/))")
            st.write("[https://www.reddit.com/r/malaysia/comments/1fsqo2j/can_nonmuslims_give_zakat](https://www.reddit.com/r/malaysia/comments/1fsqo2j/can_nonmuslims_give_zakat)")
 
        if st.button("Generate Report"):
            if reddit_url:
                if st.session_state['previous_link'] != reddit_url:
                    generate_report.clear()  # Clear cached data for new URL
                    st.session_state['previous_link'] = reddit_url
                    st.session_state['report'] = None

                submission_id = get_submission_id(reddit_url)
                if submission_id:
                    title, body, author = fetch_reddit_post(submission_id)
                    if title and body and author:
                        # Generate the report
                        try:
                            report = generate_report(title, body, author, reddit_url)
                            st.session_state['report'] = report
                            st.success("Report generated successfully! Navigate to 'Full Report' or other sections using the sidebar.")
                        except Exception as e:
                            st.error(f"An error occurred during report generation: {e}")
                            return

                        # Write report to the template
                        try:
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            template_path = os.path.join(current_dir, 'template.xlsx')
                            excel_data = write_report_to_excel(report, template_path)
                        except Exception as e:
                            st.error(f"Error writing report to Excel: {e}")
                            excel_data = None

                        if excel_data:
                            # Create a unique filename
                            safe_title = ''.join(c for c in report['Title'].replace(" ", "_")[:20] if c.isalnum() or c == '_')
                            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            filename = f"{safe_title}-{date_str}.xlsx"

                            st.download_button(
                                label="Download Report as Excel",
                                data=excel_data,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.error("Failed to generate the Excel report.")
                    else:
                        st.error("Could not extract post content. Please ensure the Reddit URL is correct and accessible.")
                else:
                    st.error("Invalid Reddit URL. Please provide a valid Reddit post URL.")
            else:
                st.error("Please provide a Reddit URL.")

    elif selection == "Full Report":
        # Ensure a report has been generated
        if st.session_state['report']:
            report = st.session_state['report']
            st.header("Full Reddit Incident Analysis Report")

            # Recommendations
            st.subheader("Recommendations")
            risk_level = report["Risk Level"]
            # Color mapping for risk levels
            risk_info = {
                'HIGH': {
                    'color': 'red',
                    'advice': "Take immediate action to mitigate risks."
                },
                'MEDIUM': {
                    'color': 'orange',
                    'advice': "Monitor the situation and prepare contingency plans."
                },
                'LOW': {
                    'color': 'green',
                    'advice': "Proceed with standard procedures."
                }
            }
            info = risk_info.get(risk_level.upper(), {
                'color': 'black',
                'advice': "No specific advice available for this risk level."
            })
            risk_color = info['color']
            advice = info['advice']
            # Display the Risk Level with color
            if risk_level.upper() == 'HIGH':
                st.error(f"**Risk Level:** {risk_level}")
            elif risk_level.upper() == 'MEDIUM':
                st.warning(f"**Risk Level:** {risk_level}")
            elif risk_level.upper() == 'LOW':
                st.success(f"**Risk Level:** {risk_level}")
            else:
                st.info(f"**Risk Level:** {risk_level}")

            # Display the Advice
            st.markdown(f"**Advice:** {advice}")
            st.write(report["Summary"])

            # Detailed Analysis
            st.subheader("Detailed Analysis")
            metrics = [
                ("Content Type Score", report['Content Type Score'], report['Content Type Explanation']),
                ("Sentiment Score", report['Sentiment Score'], report['Sentiment Explanation']),
                ("Targeting Score", report['Targeting Score'], report['Targeting Explanation']),
                ("Context Score", report['Context Score'], report['Context Explanation']),
                ("Urgency Score", report['Urgency Score'], report['Urgency Explanation']),
            ]

            for metric_name, score, explanation in metrics:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{metric_name}:** {score}")
                with col2:
                    st.markdown(f"*Explanation:* {explanation}")
                st.write("---")

            # Problem Statement
            st.subheader("Problem Statement")
            st.markdown(f"{report.get('Problem Statement', 'N/A')}")

            # Comment Stance Analysis
            st.subheader("Comment Stance Analysis")
            agree_pct = report.get("Agree Percentage", 0)
            disagree_pct = report.get("Disagree Percentage", 0)
            neutral_pct = report.get("Neutral Percentage", 0)

            col1, col2, col3 = st.columns(3)
            col1.metric("Agree", f"{agree_pct:.2f}%")
            col2.metric("Disagree", f"{disagree_pct:.2f}%")
            col3.metric("Neutral", f"{neutral_pct:.2f}%")

            # General Consensus
            st.subheader("General Consensus")
            st.markdown(f"The general consensus among the comments is: **{report.get('General Consensus', 'N/A')}**")

            # Visual Representation using Bar Chart
            st.subheader("Stance Distribution")
            stance_data = pd.DataFrame({
                'Stance': ['Agree', 'Disagree', 'Neutral'],
                'Percentage': [agree_pct, disagree_pct, neutral_pct]
            })
            st.bar_chart(stance_data.set_index('Stance'))

            # Pie Chart Visualization
            st.subheader("Stance Distribution (Pie Chart)")
            pie_data = pd.DataFrame({
                'Stance': ['Agree', 'Disagree', 'Neutral'],
                'Percentage': [agree_pct, disagree_pct, neutral_pct]
            })
            fig = px.pie(pie_data, names='Stance', values='Percentage', title='Comment Stance Distribution')
            st.plotly_chart(fig)


            # Excel Download Button
            # Ensure correct path to template.xlsx
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                template_path = os.path.join(current_dir, 'template.xlsx')
                excel_data = write_report_to_excel(report, template_path)

                if excel_data:
                    # Create a unique filename
                    safe_title = ''.join(c for c in report['Title'].replace(" ", "_")[:20] if c.isalnum() or c == '_')
                    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{safe_title}-{date_str}.xlsx"

                    st.download_button(
                        label="Download Report as Excel",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error("Failed to generate the Excel report. Please check the template and report data.")
            except Exception as e:
                st.error(f"An error occurred while preparing the download: {e}")

        else:
            st.warning("Please generate a report first.")

    else:
        # Ensure a report has been generated
        if st.session_state['report']:
            report = st.session_state['report']

            if selection == "Recommendations":
                st.header("Recommendations")
                risk_level = report["Risk Level"]
                # Color mapping for risk levels
                risk_info = {
                    'HIGH': {
                        'color': 'red',
                        'advice': "Take immediate action to mitigate risks."
                    },
                    'MEDIUM': {
                        'color': 'orange',
                        'advice': "Monitor the situation and prepare contingency plans."
                    },
                    'LOW': {
                        'color': 'green',
                        'advice': "Proceed with standard procedures."
                    }
                }
                info = risk_info.get(risk_level.upper(), {
                    'color': 'black',
                    'advice': "No specific advice available for this risk level."
                })
                risk_color = info['color']
                advice = info['advice']
                # Display the Risk Level with color
                if risk_level.upper() == 'HIGH':
                    st.error(f"**Risk Level:** {risk_level}")
                elif risk_level.upper() == 'MEDIUM':
                    st.warning(f"**Risk Level:** {risk_level}")
                elif risk_level.upper() == 'LOW':
                    st.success(f"**Risk Level:** {risk_level}")
                else:
                    st.info(f"**Risk Level:** {risk_level}")

                # Display the Advice
                st.markdown(f"**Advice:** {advice}")
                st.write(report["Summary"])

            elif selection == "Detailed Analysis":
                st.header("Detailed Analysis")
                metrics = [
                    ("Content Type Score", report['Content Type Score'], report['Content Type Explanation']),
                    ("Sentiment Score", report['Sentiment Score'], report['Sentiment Explanation']),
                    ("Targeting Score", report['Targeting Score'], report['Targeting Explanation']),
                    ("Context Score", report['Context Score'], report['Context Explanation']),
                    ("Urgency Score", report['Urgency Score'], report['Urgency Explanation']),
                ]

                for metric_name, score, explanation in metrics:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"**{metric_name}:** {score}")
                    with col2:
                        st.markdown(f"*Explanation:* {explanation}")
                    st.write("---")

            elif selection == "Problem Statement":
                st.header("Problem Statement")
                st.markdown(f"{report.get('Problem Statement', 'N/A')}")

            elif selection == "Comment Stance Analysis":
                st.header("Comment Stance Analysis")
                agree_pct = report.get("Agree Percentage", 0)
                disagree_pct = report.get("Disagree Percentage", 0)
                neutral_pct = report.get("Neutral Percentage", 0)

                col1, col2, col3 = st.columns(3)
                col1.metric("Agree", f"{agree_pct:.2f}%")
                col2.metric("Disagree", f"{disagree_pct:.2f}%")
                col3.metric("Neutral", f"{neutral_pct:.2f}%")

                # General Consensus
                st.subheader("General Consensus")
                st.markdown(f"The general consensus among the comments is: **{report.get('General Consensus', 'N/A')}**.")

                # Visual Representation using Bar Chart
                st.subheader("Stance Distribution")
                stance_data = pd.DataFrame({
                    'Stance': ['Agree', 'Disagree', 'Neutral'],
                    'Percentage': [agree_pct, disagree_pct, neutral_pct]
                })
                st.bar_chart(stance_data.set_index('Stance'))

                # Pie Chart Visualization
                st.subheader("Stance Distribution (Pie Chart)")
                pie_data = pd.DataFrame({
                    'Stance': ['Agree', 'Disagree', 'Neutral'],
                    'Percentage': [agree_pct, disagree_pct, neutral_pct]
                })
                fig = px.pie(pie_data, names='Stance', values='Percentage', title='Comment Stance Distribution')
                st.plotly_chart(fig)
        else:
            st.warning("Please generate a report first.")

if __name__ == "__main__":
    main()
