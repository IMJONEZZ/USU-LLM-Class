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
I tried with a more complicated model, but it took too long to run. im talking over an hour. so i went back to last weeks model.

But it turns out wikipedia has a super convenient API that allows you to search using it. so i used that as a simple agent.

If i had more time i could have made the model only reach out to wikipedia if it was unsure of the answer.

Look at the responses though:

The bottom response is with the model reaching out to wikipedia. all the others are just raw model responses.
Night and day difference.

![Demo reponse.jpg](Demo%20reponse.jpg)

