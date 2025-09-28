KeyError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/ai-applciation-course/Ai Course.py", line 21, in <module>
    forecast_pipeline = pipeline(
        "time-series-forecasting",
        model="amazon/chronos-t5-small",
        use_auth_token=HF_TOKEN
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/transformers/pipelines/__init__.py", line 966, in pipeline
    normalized_task, targeted_task, task_options = check_task(task)
                                                   ~~~~~~~~~~^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/transformers/pipelines/__init__.py", line 536, in check_task
    return PIPELINE_REGISTRY.check_task(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/transformers/pipelines/base.py", line 1549, in check_task
    raise KeyError(
        f"Unknown task {task}, available tasks are {self.get_supported_tasks() + ['translation_XX_to_YY']}"
    )
