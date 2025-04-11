# Assignment 9

## Quick Run
Make a file called keys.json in the same directory as the script. The file should look like this:
```
{
    "huggingfaceToken":"<your huggingface token>"
}
```

Then use startup.bat to run the script. This should take a full 5 minutes to run.

go to http://localhost:8000/ to see the page. Expect a 90 second wait for any question to be answered.

## Description
I love the look of streamlit, I struggled finding the right commands to control it. But after some tinkering, i was able to get the startup.bat file to run it properly.

It looks so good and the code is very clean as apposed to the vibe coded frontend i had working before. Now i can get rid of those files and just use streamlit's baked in functionality.

When running, know that it should take a full 3 minutes for the model to load up and all the required libraries to be imported. After that, it should be a smooth experience.

![streamlit_working.jpg](streamlit%working.jpg)

