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

`
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
`
