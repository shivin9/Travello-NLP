# Travello-NLP
This automation API is used by the Travello-extension to automate the extraction of names, addresses, write-up and the image-url of points of interest on a webpage.
It is used as follows:-

1) Create a parameters dictionary as follows:-

            paramsold = {'BATCH_SIZE': 512,
                'GRAD_CLIP': 100,
                'LEARNING_RATE': 0.01,
                'NAME': 'RNN',
                'NUM_CLUST': 3,
                'NUM_EPOCHS': 20,
                'NUM_FEATURES': 8,
                'N_HIDDEN': 512,
                'SEQ_LENGTH': 1,
                'TYPE': '1st classifier'}

2) Initialize the model using the parameters defined above.

            rnnModelold = getModel(paramsold, "rnnmodel-old")

3) Retrieve the addresses using the following method:-

            addresses = getAddress(url, [(paramsold, rnnModelold)])

4) Retrieve all the titles as follows:-
            
            url = 'https://sf.eater.com/maps/10-top-spots-for-indian-in-san-francisco/gajalee'
            titles = getTitle(url, addresses)

5) Retrieve all the images as follows:-

            images = getImg(url)

6) Finally consolidate all the pieces of information to create a blob
   of information using the following command:-

           finalThing = consolidateStuff(url, titles, addresses, images)

It should look something like this:-

```
{
    "1": {
        "Write-up": "This Mission spot has a coastal seafood twist, setting it far apart from other Indian in the area. Lookout for the fish malvani masala, thali and curry leaf calamari, and wash it all down with a rose lassi.",
        "Image URL": "https://cdn2.vox-cdn.com/uploads/chorus_image/image/38785334/gajalee.0.jpg",
        "Place Name": "1 Gajalee",
        "Address": [
            "525 Valencia St San Francisco, CA 94110",
            "(415) 552-9000"
        ]
    },
    
    "2": {
        "Write-up": "Prized for its ample servings of affordable vegetarian fare, Udupi Palace has developed a following of loyal dosa lovers. This South Indian spot specializes in delicious dosas, but there's also a full complement of uthappam and other classics like aloo mutter and biryani.",
        "Image URL": "https://cdn1.vox-cdn.com/uploads/chorus_image/image/38785336/udupi.0.jpg",
        "Place Name": "2 Udupi Palace",
        "Address": [
            "1007 Valencia St San Francisco, CA 94110",
            "(415) 970-8000"
        ]
    }
}
```

# So, How does it actually work?
The web is filled with reviews of travel spots, dining places, historical sites etc. in form of structured, professional resources like **[TripAdvisor](https://www.tripadvisor.in/)** or **[MakeMyTrip](https://www.makemytrip.com/)** and unstructured resources (mostly blogs) like **[LadyIronChef](http://www.ladyironchef.com/)** or **[Boston Eater](https://boston.eater.com/)**. Making an application which can mine relevant, structured information from all these varieties of places is surely a challenge but in this project we have managed to acheive a decent performance in terms of precision, recall and versitality.

## Mining data from structured resources.
This was pretty easy to say the least. One has to just mine the html page and look for specific, named tags which store the required information. eg. on TripAdvisor all the headings are in the `<span>` tag while addresses are nicely encapsulated in the `<street-address>`, `<locality>` and the `<country-name>` tags.

## Mining data from blogs
This was the really challenging part of the project which took the major time of my internship. There are two major problems which prohibit us from employing a generic solution by seeing the html structure of the webpage:-

* People use different blogging services to create their blogs which makes sure that every blog is a different website altogether with different ways of storing data usually blog posts

* Every blog writer has a different style of writing and arranging their page which makes the task of our information scraper even more difficult. Also, unlike professional websites like **TripAdvisor**, blogs are usually simple and everything is written in form of plain text

These factors motivated me give up on html-page mining approach and I turned to machine learning for help. The only hope of success of such a solution lied in the fact that even though the html-pages of blogs are unique, every blog post follows a loosely similar structure ie. every post has a **Main Title** eg. 10 best restaurants in Singapore followed by details of 10 restaurants which are themselves structured having a **Title**, **Text body**, **Photo** and **Address**.

So I tried to capture any one of these parameters and then mine the other three as they lie nearby only. Title names can be quite awkward and hard to classify therefore I went forward with trying to classify addresses to a get loose hint of exactly where the places of interest might lie on a page and then extract other relevant information.

# Identifying addresses on a page
Even though addresses show some structure within themselves eg. having **house name**, **street name**, **locality**, **city**, **state** and **country** followed by a **phone number** sometimes, but still not every piece of information is present all the time. The only thing that we can depend on is that the order of features remains the same ie. country name never follows the street name. The relative sequencing of the features remains the same.

This motivated me to use a Recurrent Neural Network as they can have a dynamic temporal memory which can process arbitrary sequences of inputs.

# Where did the training data come from?
Getting the training data was another big hurdle as there is no repository of all the addresses or place names in the world. Neither we have a collection of annotated web pages where text and addresses are marked.
(If only we had a mechanism to mark addresses in a piece of text then we could get the training data... but then ofcourse we wouldn't need this project in the first place!)

So finally it struck me that if I find can't a repository of blog posts with text and addressed marked then I can simply make one!

This turned out to be quite easy as I just had to select random sentences from [here](https://github.com/shivin9/Travello-NLP/blob/master/database/hard_data/garbage) and sprinkle them with partially structures addresses (how they would occur in real life blog posts) chosen from [here](https://github.com/shivin9/Travello-NLP/tree/master/database/hard_data).

Thus I trained my RNN on around 11,000 blog posts by **feeding 5 lines** at a time (2 lines from past, 1 from present and 2 from future) so that the classifier can learn how to exactly figure out an address amidst pieces of text.

# Features
The feature set for a sentence is simple, there are 9 feature defined [here](https://github.com/shivin9/Travello-NLP/blob/master/create_training.py#L277)


# Putting it all together
After we parse a page and get the location of addresses (in form of line numbers) we use this information to extract the name of place as well. Knowing that above the address lies a paragraph or two of **text description**, maybe a **photo** and then the **title** we capture this information using a few simple heuristic arguments on the locality of addresses, titles etc.

For implementation details you can see the code and ping me in case of doubts!

# Using Travello-NLP

1) Make sure that you have all the libraries given in the requirements
2) Enable GPU for theano for better performance
3) Pre-trained models are stored in the `./models/` folder. You can train your own by chaning the parameter in the `portal.py` file.
4) To debug the application you can either execute each function independently in iPython or uncomment [these](https://github.com/shivin9/Travello-NLP/blob/master/portal.py#L181-L185) lines.
5) Run the application using `python portal.py`

=======
