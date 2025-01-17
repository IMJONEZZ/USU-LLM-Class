# Read Me

This the read me file for Riley May's branch for assignment 2. 

### Answer to question 4. 

I did not do a lot to improve the tokenizer because I didn't have an end goal of tokenizer. If I was doing something like topic modeling, I would have probably removed stop words or words with really low frequencies. 

The two big improvements I made were with regard to contractions and capitalization. I used a short, but effective list of contractions. I separated all these words into  two words because I didn't want the word I'm to be split into three tokens I ' m with the current tokenizer. This will preserve meaning while also not having non sense words like "ve" from I've.  Also, I made all the words lower case because the word "stormtrooper" should be the same token regardless if it is at the beginning or middle of a sentence. 
 

These improvements took it from from about 3250 words in my dictionary to 2800. 
