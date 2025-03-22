# Assignment 8 

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
This script serves up last week's assignment in a web page. I was going to use flask to do it (im most familiar with it), but after reading the given instructions I decided to use fastapi instead.
It works almost exactly the same as flask, honestly, I i could have stuck with flask and it would have been fine.

This script uses the super simple vectorDB to answer a question. You will see if it uses a snipped from vector db because it doesn't take 90 seconds to get an answer to you. also it will say if it uses the built in db.

