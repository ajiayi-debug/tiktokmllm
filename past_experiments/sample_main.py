"""
Main script to orchestrate the AI video analysis pipeline.

This script runs a sequence of agents:
1. ContextRetrieverAgent: Extracts context from videos based on questions.
2. RecommendedFormatPromptAgent: Formats the context and questions using a template.
3. CotAgent: Performs Chain-of-Thought reasoning to answer questions based on formatted context.
"""

import asyncio
import logging
import argparse
import pandas as pd
from pathlib import Path
import time
import os
import json
import sys

# --- Agent Imports ---
# Try importing agents and provide clear error messages if they fail.
def import_agent(module_path, class_name, is_function=False):
    try:
        module = __import__(module_path, fromlist=[class_name])
        agent = getattr(module, class_name)
        logger.info(f"Successfully imported {'function' if is_function else 'class'} '{class_name}' from '{module_path}'")
        return agent
    except ImportError as e:
        logger.error(f"ImportError: Could not import '{class_name}' from '{module_path}'. Check path and module contents. Error: {e}")
        return None
    except AttributeError:
        logger.error(f"AttributeError: Could not find '{class_name}' within '{module_path}'. Check the class/function name.")
        return None

ContextRetrieverAgent = import_agent("agents.context_retriever_agent.agent", "ContextRetrieverAgent")
run_format_prompt_agent_main = import_agent("agents.recommended_format_prompt_agent", "main", is_function=True)
CotAgent = import_agent("agents.cot_agent.gemini_cot_agent", "CotAgent")

# --- Logging Setup ---
# Configure root logger for file and console output
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File Handler
log_file_handler = logging.FileHandler("pipeline.log")
log_file_handler.setFormatter(log_formatter)

# Console Handler
log_console_handler = logging.StreamHandler(sys.stdout)
log_console_handler.setFormatter(log_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Set root level
root_logger.addHandler(log_file_handler)
root_logger.addHandler(log_console_handler)

# Get a specific logger for this module
logger = logging.getLogger("PipelineOrchestrator")

# --- Configuration Class ---
class PipelineConfig:
    """Stores and validates pipeline configuration and paths."""
    def __init__(self, args):
        self.args = args
        self._validate_paths()
        self._setup_paths()

    def _validate_paths(self):
        # Validate essential input paths immediately
        self.initial_data_path = Path(self.args.input_csv)
        if not self.initial_data_path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {self.initial_data_path}")

        self.format_template_path_arg = Path(self.args.format_template_path)
        self.cot_iterate_template_path_arg = Path(self.args.cot_iterate_template_path)

        if not self.format_template_path_arg.exists() and not self.format_template_path_arg.is_absolute():
             # Assume relative to script dir if not found directly
             self.format_template_path = Path(__file__).parent / self.format_template_path_arg
        else:
            self.format_template_path = self.format_template_path_arg
        if not self.format_template_path.is_file():
             raise FileNotFoundError(f"Formatting template not found: {self.format_template_path}")

        if not self.cot_iterate_template_path_arg.exists() and not self.cot_iterate_template_path_arg.is_absolute():
            # Assume relative to script dir
            self.cot_iterate_template_path = Path(__file__).parent / self.cot_iterate_template_path_arg
        else:
            self.cot_iterate_template_path = self.cot_iterate_template_path_arg
        if not self.cot_iterate_template_path.is_file():
            raise FileNotFoundError(f"CoT iteration template not found: {self.cot_iterate_template_path}")

    def _setup_paths(self):
        # Create a run-specific output directory
        self.run_output_dir = Path(self.args.output_base_dir) / f"run_{self.args.run_suffix}_{int(time.time())}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run-specific output directory created: {self.run_output_dir}")

        # --- Define paths for data flowing between agents ---

        # Context Retriever Agent paths (outputs usually go to data/ and checkpoints/)
        self.context_agent_final_suffix = f"ContextRetriever_{self.args.run_suffix}"
        self.context_output_path = Path("data") / f"Step1_Context_{self.context_agent_final_suffix}.json"
        self.context_initial_chkpt_suffix = f"ContextRetriever_initial_{self.args.run_suffix}"
        self.context_retry_chkpt_suffix = f"ContextRetriever_retry_{self.args.run_suffix}"

        # Formatting Agent paths (output goes to run-specific dir)
        self.formatted_output_path = self.run_output_dir / f"step2_formatted_output_{self.args.run_suffix}.json"

        # Cot Agent paths (outputs usually go to data/ and checkpoints/)
        self.cot_initial_checkpoint_suffix = f"step3_cot_{self.args.run_suffix}_initial"
        self.cot_retry_checkpoint_suffix = f"step3_cot_{self.args.run_suffix}_retry"
        # Expected final CSV output path from CotAgent's reorder step
        self.cot_final_output_path = Path("data") / f"{self.cot_initial_checkpoint_suffix}_rearranged.csv"

        logger.info(f"Input Data Path: {self.initial_data_path}")
        logger.info(f"Expected Context Output (Step 1): {self.context_output_path}")
        logger.info(f"Expected Formatted Output (Step 2): {self.formatted_output_path}")
        logger.info(f"Expected CoT Final Output (Step 3): {self.cot_final_output_path}")

    # Provide easy access to underlying args
    def __getattr__(self, name):
        if hasattr(self.args, name):
            return getattr(self.args, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# --- Orchestrator Class ---
class PipelineOrchestrator:
    """Manages the execution flow of the AI pipeline steps."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info(f"PipelineOrchestrator initialized with run suffix: {config.run_suffix}")

    async def run_context_retrieval(self):
        """Executes Step 1: Context Retrieval."""
        logger.info("--- Starting Step 1: Context Retrieval ---")
        step_start_time = time.time()

        if not ContextRetrieverAgent:
            raise ImportError("ContextRetrieverAgent unavailable, cannot run Step 1.")

        try:
            df = pd.read_csv(self.config.initial_data_path)
            logger.info(f"Calling ContextRetrieverAgent for {len(df)} items. Expecting output: {self.config.context_output_path}")

            # Call the agent's main async function (assuming it returns the final path)
            actual_output_path_str = await ContextRetrieverAgent(
                df=df,
                checkpoint_path_initial=self.config.context_initial_chkpt_suffix,
                checkpoint_path_retry=self.config.context_retry_chkpt_suffix,
                final_output_suffix=self.config.context_agent_final_suffix,
                temperature=self.config.context_temperature,
                video_upload=self.config.video_upload,
                wait_time=self.config.context_wait_time,
                concurrency_limit=self.config.context_concurrency,
            )

            actual_output_path = Path(actual_output_path_str)
            # Handle potential discrepancy between expected and actual path
            if actual_output_path != self.config.context_output_path:
                 logger.warning(f"ContextRetrieverAgent returned path '{actual_output_path}' which differs from expected '{self.config.context_output_path}'. Updating expected path.")
                 self.config.context_output_path = actual_output_path # Update config for next step

            # Validate output existence
            if not self.config.context_output_path.exists():
                 raise FileNotFoundError(f"Context output missing after Step 1: {self.config.context_output_path}")

            step_duration = time.time() - step_start_time
            logger.info(f"--- Step 1 completed in {step_duration:.2f}s. Output: {self.config.context_output_path} ---")
            return self.config.context_output_path
        except Exception as e:
            logger.error(f"Step 1 (Context Retrieval) failed: {e}", exc_info=True)
            raise # Re-raise to halt pipeline

    async def run_formatting(self):
        """Executes Step 2: Prompt Formatting."""
        logger.info("--- Starting Step 2: Prompt Formatting ---")
        step_start_time = time.time()

        if not run_format_prompt_agent_main:
             raise ImportError("Formatting agent main function unavailable, cannot run Step 2.")

        # Check prerequisite file from Step 1
        if not self.config.context_output_path.exists():
            raise FileNotFoundError(f"Input for Step 2 missing: {self.config.context_output_path}")

        try:
            logger.info(f"Calling formatting agent. Input: {self.config.context_output_path}, Template: {self.config.format_template_path}, Output: {self.config.formatted_output_path}")

            # Call the formatting agent's main async function
            await run_format_prompt_agent_main(
                 input_path_str=str(self.config.context_output_path),
                 output_path_str=str(self.config.formatted_output_path),
                 template_path_str=str(self.config.format_template_path),
                 concurrency=self.config.format_concurrency,
                 max_retries=self.config.format_max_retries,
                 initial_backoff=self.config.format_initial_backoff
            )

            # Validate output existence
            if not self.config.formatted_output_path.exists():
                 raise FileNotFoundError(f"Formatted output missing after Step 2: {self.config.formatted_output_path}")

            step_duration = time.time() - step_start_time
            logger.info(f"--- Step 2 completed in {step_duration:.2f}s. Output: {self.config.formatted_output_path} ---")
            return self.config.formatted_output_path
        except Exception as e:
            logger.error(f"Step 2 (Prompt Formatting) failed: {e}", exc_info=True)
            raise # Re-raise to halt pipeline

    def _prepare_cot_input(self) -> pd.DataFrame:
        """Loads and merges data required for the CoT Agent."""
        logger.info("Preparing data for CoT Agent...")

        # Check prerequisite files
        if not self.config.initial_data_path.exists():
             raise FileNotFoundError(f"Original input CSV missing: {self.config.initial_data_path}")
        if not self.config.formatted_output_path.exists():
             raise FileNotFoundError(f"Formatted data from Step 2 missing: {self.config.formatted_output_path}")

        # Load original data
        logger.debug(f"Loading original data: {self.config.initial_data_path}")
        df_original = pd.read_csv(self.config.initial_data_path, keep_default_na=False)

        # Load formatted data
        logger.debug(f"Loading formatted data: {self.config.formatted_output_path}")
        with open(self.config.formatted_output_path, 'r', encoding='utf-8') as f:
             formatted_data = json.load(f)
        if not isinstance(formatted_data, list) or not formatted_data or 'qid' not in formatted_data[0]:
            raise ValueError(f"Invalid or empty formatted data structure in {self.config.formatted_output_path}")
        df_formatted = pd.DataFrame(formatted_data)

        # Select necessary columns from formatted data
        cols_to_merge = ['qid']
        if 'corrected_question' in df_formatted.columns:
             cols_to_merge.append('corrected_question')
        else:
             logger.warning("'corrected_question' missing in formatted data. CoT will use original 'question'.")
        if 'context' in df_formatted.columns:
             cols_to_merge.append('context')
        else:
             raise ValueError(f"Required 'context' column missing in formatted data: {self.config.formatted_output_path}")

        df_formatted_subset = df_formatted[cols_to_merge].copy()
        # Ensure qid is string for merge robustness
        df_formatted_subset['qid'] = df_formatted_subset['qid'].astype(str)
        df_original['qid'] = df_original['qid'].astype(str)

        # Merge
        logger.info("Merging original data with formatted data on 'qid' for CoT input.")
        df_merged = pd.merge(df_original, df_formatted_subset, on='qid', how='left')

        # Post-merge handling: Fill missing corrected_question, check context
        if 'corrected_question' in df_merged.columns:
            missing_cq_mask = df_merged['corrected_question'].isnull()
            df_merged.loc[missing_cq_mask, 'corrected_question'] = df_merged.loc[missing_cq_mask, 'question']
            logger.info(f"Filled {missing_cq_mask.sum()} missing 'corrected_question' values from original 'question'.")
        else:
            df_merged['corrected_question'] = df_merged['question'] # Create if it doesn't exist

        if df_merged['context'].isnull().any():
             missing_context_qids = df_merged[df_merged['context'].isnull()]['qid'].tolist()
             logger.error(f"CRITICAL: Context is missing for QIDs after merge: {missing_context_qids}")
             raise ValueError(f"Context missing for QIDs: {missing_context_qids}")

        logger.info(f"Data preparation for CoT complete. Merged DataFrame shape: {df_merged.shape}")
        return df_merged

    async def run_cot(self):
        """Executes Step 3: CoT Agent Execution."""
        logger.info("--- Starting Step 3: CoT Agent Execution ---")
        step_start_time = time.time()

        if not CotAgent:
             raise ImportError("CotAgent unavailable, cannot run Step 3.")

        try:
            # Prepare the input DataFrame
            df_cot_input = self._prepare_cot_input()

            # Load and format the iteration prompt template
            try:
                template = self.config.cot_iterate_template_path.read_text(encoding="utf-8").strip()
                iterate_prompt = template.format(iteration=self.config.cot_iterations)
                logger.debug(f"Using CoT iteration prompt template from: {self.config.cot_iterate_template_path}")
            except Exception as e:
                 raise ValueError(f"Failed to load/format CoT iteration prompt template: {self.config.cot_iterate_template_path} - {e}")

            logger.info(f"Calling CotAgent with {len(df_cot_input)} items. Expecting final output: {self.config.cot_final_output_path}")

            # Call CotAgent
            await CotAgent(
                 df=df_cot_input,
                 checkpoint_path_initial=self.config.cot_initial_checkpoint_suffix,
                 checkpoint_path_retry=self.config.cot_retry_checkpoint_suffix,
                 number_of_iterations=1, # Seems fixed in agent?
                 temperature=self.config.cot_temperature,
                 iterate_prompt=iterate_prompt,
                 video_upload=self.config.video_upload,
                 wait_time=self.config.cot_wait_time,
                 iteration_in_prompt=self.config.cot_iterations # Pass the intended number
            )

            # Validate expected output file
            if not self.config.cot_final_output_path.exists():
                 logger.warning(f"CoT Agent finished, but expected final rearranged CSV is missing: {self.config.cot_final_output_path}. Check agent's internal logic.")
            else:
                 logger.info(f"CoT Agent finished. Final output expected at: {self.config.cot_final_output_path}")

            step_duration = time.time() - step_start_time
            logger.info(f"--- Step 3 completed in {step_duration:.2f}s. --- ")
            # No return value needed, side effect is the final CSV file

        except Exception as e:
            logger.error(f"Step 3 (CoT Agent) failed: {e}", exc_info=True)
            raise # Re-raise to halt pipeline

    async def run_pipeline(self):
        """Runs the full pipeline sequentially."""
        logger.info("====== Starting AI Pipeline Orchestration ======")
        pipeline_start_time = time.time()
        final_status = "FAILED"
        try:
            await self.run_context_retrieval()
            await self.run_formatting()
            await self.run_cot()
            final_status = "SUCCESS"
            logger.info(f"Successfully completed all pipeline steps.")

        except (FileNotFoundError, ImportError, ValueError) as e:
             logger.critical(f"Pipeline halted due to configuration or execution error: {e}", exc_info=False) # Log specifics, but not full trace usually
        except Exception as e:
            logger.critical(f"Pipeline halted due to unexpected error: {e}", exc_info=True) # Log full trace for unexpected errors
        finally:
            pipeline_duration = time.time() - pipeline_start_time
            logger.info(f"====== AI Pipeline Finished - Status: {final_status} ======")
            logger.info(f"Total execution time: {pipeline_duration:.2f} seconds.")
            logger.info(f"Run-specific outputs (e.g., formatted data) are in: {self.config.run_output_dir}")
            logger.info(f"Agent outputs (context, final predictions) are likely in: data/ and checkpoints/")
            logger.info(f"See full details in log file: pipeline.log")

# --- Argument Parsing Function ---
def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrate AI Video Analysis Pipeline")

    # --- General ---
    pg_general = parser.add_argument_group('General Pipeline Options')
    pg_general.add_argument("--input_csv", type=str, default="data/data.csv", help="Path to the input CSV data file. Default: data/data.csv")
    pg_general.add_argument("--output_base_dir", type=str, default="pipeline_runs", help="Base directory for pipeline run-specific outputs. Default: pipeline_runs")
    pg_general.add_argument("--run_suffix", type=str, default="default_run", help="Suffix for naming checkpoints and final outputs for this run. Default: default_run")
    pg_general.add_argument("--video_upload", action='store_true', help="Upload local videos instead of using URLs (applies to all relevant agents).")

    # --- Context Retriever (Step 1) ---
    pg_step1 = parser.add_argument_group('Step 1: Context Retriever Options')
    pg_step1.add_argument("--context_temperature", type=float, default=0.0, help="Temperature for Context Retriever generation. Default: 0.0")
    pg_step1.add_argument("--context_wait_time", type=int, default=10, help="Wait time (seconds) between Context Retriever API calls. Default: 10")
    pg_step1.add_argument("--context_concurrency", type=int, default=10, help="Concurrency limit for Context Retriever internal tasks. Default: 10")

    # --- Formatter (Step 2) ---
    pg_step2 = parser.add_argument_group('Step 2: Prompt Formatter Options')
    pg_step2.add_argument("--format_template_path", type=str, default="templates/prompt_template.txt", help="Path to the prompt formatting template file. Default: templates/prompt_template.txt")
    pg_step2.add_argument("--format_concurrency", type=int, default=20, help="Concurrency limit for Formatter API calls. Default: 20")
    pg_step2.add_argument("--format_max_retries", type=int, default=3, help="Max retries for Formatter API calls. Default: 3")
    pg_step2.add_argument("--format_initial_backoff", type=float, default=2.0, help="Initial backoff (seconds) for Formatter retries. Default: 2.0")

    # --- CoT Agent (Step 3) ---
    pg_step3 = parser.add_argument_group('Step 3: CoT Agent Options')
    pg_step3.add_argument("--cot_temperature", type=float, default=0.0, help="Temperature for CoT Agent generation. Default: 0.0")
    pg_step3.add_argument("--cot_iterations", type=int, default=8, help="Number of candidate answers for CoT Agent generation. Default: 8")
    pg_step3.add_argument("--cot_iterate_template_path", type=str, default="templates/iterate_prompt.txt", help="Path to CoT iteration prompt template file. Default: templates/iterate_prompt.txt")
    pg_step3.add_argument("--cot_wait_time", type=int, default=10, help="Wait time (seconds) between CoT Agent API calls. Default: 10")

    return parser.parse_args()

# --- Main Execution Block ---
async def run():
    # Check essential imports before parsing args or initializing
    if not all([ContextRetrieverAgent, run_format_prompt_agent_main, CotAgent]):
         logger.critical("One or more required agent components could not be imported. Pipeline cannot start.")
         print("\nCRITICAL ERROR: Failed to import one or more essential agent components.")
         print("Please check the import statements and file paths at the top of main.py and ensure the agent files exist and are correct.")
         print("See pipeline.log for specific import errors.")
         sys.exit(1)

    args = parse_args()
    try:
        config = PipelineConfig(args)
        orchestrator = PipelineOrchestrator(config=config)
        await orchestrator.run_pipeline()
    except FileNotFoundError as e:
        logger.critical(f"Pipeline initialization failed: {e}")
        print(f"\nCRITICAL ERROR: A required file was not found during setup. Check paths. Error: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch broad exceptions during setup or execution not caught by run_pipeline
        logger.critical(f"Unhandled exception during pipeline setup or execution: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: Pipeline failed unexpectedly during setup or run. Check pipeline.log.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run()) 