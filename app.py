import streamlit as st
from services.video_processing_service import VideoProcessingService
import asyncio
import os
from dotenv import load_dotenv

load_dotenv() 

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEY not found. Please ensure it is set in your .env file in the project root.")
    st.stop()

video_service = VideoProcessingService(gemini_api_key=gemini_api_key)

st.set_page_config(layout="wide")
st.title("🎬 Video Analysis: Gemini vs. Agentic Workflow")

live_progress_placeholder = st.empty()

st.sidebar.header("Inputs")
video_url = st.sidebar.text_input("Video URL:", placeholder="Enter YouTube or other video link")
user_question = st.sidebar.text_area("Your Question:", placeholder="e.g., What did the capybara do?", height=150)

st.sidebar.markdown("##### Number of candidate answers:")

if 'selected_candidate_option' not in st.session_state:
    st.session_state.selected_candidate_option = 8 # Default to 8
if 'custom_candidate_num' not in st.session_state:
    st.session_state.custom_candidate_num = st.session_state.selected_candidate_option

predefined_options_list = [4, 8, 12, 16, 20, 24, 28, 32]
custom_option_label = "Custom..."
options_for_selectbox = predefined_options_list + [custom_option_label]

try:
    if st.session_state.selected_candidate_option == custom_option_label:
        current_selectbox_index = options_for_selectbox.index(custom_option_label)
    else:
        current_selectbox_index = options_for_selectbox.index(int(st.session_state.selected_candidate_option))
except ValueError:
    st.session_state.selected_candidate_option = 8 
    current_selectbox_index = options_for_selectbox.index(8)

selected_option = st.sidebar.selectbox(
    "Select or choose 'Custom...':",
    options=options_for_selectbox,
    index=current_selectbox_index,
    key='selectbox_master_key',
    help="Select a predefined number or choose 'Custom...' to enter your own."
)

st.session_state.selected_candidate_option = selected_option 

num_candidate_answers_to_use = 0
if selected_option == custom_option_label:
    custom_num = st.sidebar.number_input(
        "Enter custom number:",
        min_value=1,
        max_value=50,
        value=st.session_state.custom_candidate_num, 
        step=1,
        key='custom_num_input_key'
    )
    st.session_state.custom_candidate_num = custom_num 
    num_candidate_answers_to_use = custom_num
elif selected_option is not None: 
    num_candidate_answers_to_use = int(selected_option)
    st.session_state.custom_candidate_num = num_candidate_answers_to_use 
else: 
    num_candidate_answers_to_use = 8 

submit_button = st.sidebar.button("Analyze & Compare")

if 'progress_log' not in st.session_state:
    st.session_state.progress_log = []

if 'gemini_alone_ans' not in st.session_state:
    st.session_state.gemini_alone_ans = ""
if 'gemini_alone_thoughts' not in st.session_state:
    st.session_state.gemini_alone_thoughts = None

if 'agentic_workflow_ans' not in st.session_state:
    st.session_state.agentic_workflow_ans = ""
if 'agentic_workflow_thoughts' not in st.session_state:
    st.session_state.agentic_workflow_thoughts = None

if 'processing' not in st.session_state:
    st.session_state.processing = False

# for flashing effect
COLOR_SHADE_1 = "#888888" 
COLOR_SHADE_2 = "#AAAAAA" 

current_pulsing_task = None
stop_pulse_event = asyncio.Event()

async def run_analysis_async(v_url, u_question, num_candidates_val, progress_placeholder): 
    global current_pulsing_task, stop_pulse_event
    st.session_state.processing = True
    st.session_state.gemini_alone_ans = "Processing..."
    st.session_state.gemini_alone_thoughts = None
    st.session_state.agentic_workflow_ans = "Processing..."
    st.session_state.agentic_workflow_thoughts = None
    
    st.session_state.progress_log = [] 
    
    async def continuous_pulse_loop(message, placeholder, stop_event_obj, pulse_delay=0.4):
        """Continuously pulses the color of a message until stop_event_obj is set."""
        while not stop_event_obj.is_set():
            try:
                placeholder.markdown(f"<p style='color: {COLOR_SHADE_1};'>{message}</p>", unsafe_allow_html=True)
                await asyncio.wait_for(stop_event_obj.wait(), timeout=pulse_delay)
            except asyncio.TimeoutError:
                pass 
            if stop_event_obj.is_set(): break 

            try:
                placeholder.markdown(f"<p style='color: {COLOR_SHADE_2};'>{message}</p>", unsafe_allow_html=True)
                await asyncio.wait_for(stop_event_obj.wait(), timeout=pulse_delay)
            except asyncio.TimeoutError:
                pass 
            if stop_event_obj.is_set(): break 

    async def update_progress_and_start_pulsing(message, main_step_delay_after=0.1):
        global current_pulsing_task, stop_pulse_event

        st.session_state.progress_log.append(message)

        # Signal previous pulsing task to stop, if any
        if current_pulsing_task:
            # print(f"Stopping previous pulse task for: {current_pulsing_task.get_name()}") # Debug
            stop_pulse_event.set() # Signal stop
            try:
                await asyncio.wait_for(current_pulsing_task, timeout=1.0) # Give it a moment to stop
            except asyncio.TimeoutError:
                # print("Timeout waiting for previous pulse task to stop. Cancelling.") # Debug
                current_pulsing_task.cancel() # Force cancel if it doesn't stop
            except asyncio.CancelledError:
                pass # Expected if cancelled
        
        stop_pulse_event.clear() 
        progress_placeholder.markdown(f"<p style='color: {COLOR_SHADE_1};'>{message}</p>", unsafe_allow_html=True)
        
        current_pulsing_task = asyncio.create_task(
            continuous_pulse_loop(message, progress_placeholder, stop_pulse_event),
            name=f"pulse_{message[:10]}" 
        )
        
        await asyncio.sleep(main_step_delay_after) 

    # ---- Workflow Steps ----
    try:
        await update_progress_and_start_pulsing("Analyzing video... Starting workflows.", main_step_delay_after=0.2)

        await update_progress_and_start_pulsing("Executing Gemini Alone workflow...")
        res_alone_tuple = await video_service.get_gemini_alone_response(v_url, u_question)
        
        await update_progress_and_start_pulsing(f"Agent 1: Candidate Generation Agent - Generating the top {num_candidates_val} answers...")
        res_agentic_tuple = await video_service.get_agentic_workflow_response(v_url, u_question, num_candidates_val) 
        
        await update_progress_and_start_pulsing("Agent 2: Answer Aggregation Agent - Selecting the best answer from generated candidates.", main_step_delay_after=0.5)
        
        st.session_state.progress_log.append("Processing complete. Populating results.")
        # Stop the last pulsing task before final message
        if current_pulsing_task:
            stop_pulse_event.set()
            try: await asyncio.wait_for(current_pulsing_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError): pass
        progress_placeholder.markdown(f"<p style='color: {COLOR_SHADE_1};'>Processing complete. Populating results.</p>", unsafe_allow_html=True)
        await asyncio.sleep(1) 
    finally:
        # Ensure the last pulsing task is definitely stopped on exit or error
        if current_pulsing_task:
            stop_pulse_event.set()
            try: await asyncio.wait_for(current_pulsing_task, timeout=0.5) # Short wait
            except (asyncio.TimeoutError, asyncio.CancelledError): pass # Ignore errors on cleanup
        current_pulsing_task = None
        progress_placeholder.empty()
        st.session_state.processing = False

    st.session_state.gemini_alone_ans = res_alone_tuple[0]
    st.session_state.gemini_alone_thoughts = res_alone_tuple[1]
    
    st.session_state.agentic_workflow_ans = res_agentic_tuple[0]
    st.session_state.agentic_workflow_thoughts = res_agentic_tuple[1]
    
    print(f"DEBUG: Gemini Alone Answer for UI: {st.session_state.gemini_alone_ans}, Type: {type(st.session_state.gemini_alone_ans)}")
    print(f"DEBUG: Gemini Alone Thoughts for UI: {st.session_state.gemini_alone_thoughts}, Type: {type(st.session_state.gemini_alone_thoughts)}")
    print(f"DEBUG: Agentic Workflow Answer for UI: {st.session_state.agentic_workflow_ans}, Type: {type(st.session_state.agentic_workflow_ans)}")
    print(f"DEBUG: Agentic Workflow Thoughts for UI: {st.session_state.agentic_workflow_thoughts}, Type: {type(st.session_state.agentic_workflow_thoughts)}")

if submit_button and video_url and user_question:
    if not st.session_state.processing:
        # Use the resolved num_candidate_answers_to_use
        effective_num_candidates = num_candidate_answers_to_use 

        with st.spinner("Analyzing... Please wait. This might take a moment. (Detailed progress above spinner, full log below after completion)"):
            asyncio.run(run_analysis_async(video_url, user_question, effective_num_candidates, live_progress_placeholder))
        st.rerun() 

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 Gemini 2.5 Pro (Alone)")
    st.markdown(st.session_state.gemini_alone_ans if st.session_state.gemini_alone_ans else "Awaiting input...")
    if st.session_state.gemini_alone_thoughts and st.session_state.gemini_alone_thoughts.strip() != "":
        with st.expander("View Thought Process (Gemini Alone)"):
            st.markdown(st.session_state.gemini_alone_thoughts)
    
with col2:
    st.subheader("✨ Gemini 2.5 Pro + Agentic Workflow")
    st.markdown(st.session_state.agentic_workflow_ans if st.session_state.agentic_workflow_ans else "Awaiting input...")
    if st.session_state.agentic_workflow_thoughts and st.session_state.agentic_workflow_thoughts.strip() != "":
        with st.expander("View Thought Process (Agentic Workflow)"):
            st.markdown("**Agent 1: Candidate Generation Agent:**")
            st.markdown("Generated the following candidate answers based on the video and question:") # Description for Agent 1
            st.markdown(st.session_state.agentic_workflow_thoughts) # Output of Agent 1
            st.markdown("---") 
            st.markdown("**Agent 2: Answer Aggregation Agent:**")
            st.markdown("Analyzed the candidate answers from Agent 1 and selected/synthesized the following final answer:") # Description for Agent 2
            st.markdown(st.session_state.agentic_workflow_ans) # Output of Agent 2 (the final answer)

# Processing Log Expander (after results, or viewable during if page is scrolled)
if st.session_state.progress_log:
    with st.expander("View Full Processing Log", expanded=False):
        for msg in st.session_state.progress_log:
            st.text(msg)
        if not st.session_state.processing and st.session_state.progress_log and st.session_state.progress_log[-1].startswith("Processing complete"):
             st.caption("Processing finished.")
        elif st.session_state.processing:
             st.caption("Processing in progress...")

if st.session_state.processing and not submit_button: 
     pass 