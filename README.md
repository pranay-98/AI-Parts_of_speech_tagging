

### Part-of-speech tagging

- A basic problems in Natural Language Processing is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.)
- Implementing Parts of speech Tagging in three methods:

  - Simplified Model (Implemented using Naives Bayes )
  - Hidden Markov Model (Implemented using Viterbi algorithm)
  - Complicated Model (Implemented using MCMC- Gibb's Sampling Technique)

- First Step is to train the given data:

  - Storing the occurences of word with specificed pos label using dictionary of dictionaries.
  - Calculating the transition occurences of present label from the previous label using dictionary of dictionaries.
  - Maintaing the occurences of pos labels over all the sentences in file using dictionary.

- **Naives Bayes(Simplified Model)**

  - Probability is calculated as follows:
    - $P(label|Word) = max{P(Word|label)\, P(label) / P(Word)}$
  - Initialzed an array of particular sentence size with nouns.
  - Obtaining and storing all distinct parts of speech labels from the training data in a list.
  - For each word in a sentence,
    - Evaluate the likelihood for each distinct part of speech label by multiplying the emission probability of the word provided that is this pos label by the probability of that label in training data.
    - By comparing the probability to the probability of the preceding label, the maximum probability and associated label are maintained.
    - In a simplified model, the label with the maximum probability is allocated to the word.
  - After scanning the entire sentence, the array with labels is returned.

  - If any word is not detected in the training data, this model will fail. I'm presuming it's a noun so that it can cover various scenarios.

- **HMM - Viterbi**

  - Viterbi is dynamic programming technique to solve HMM model. Here, I used table of dictionaries to store the probabilities at each state for all hidden labels.
  - For the first word in a sentence we calculate initial probability as the probability of that label and multiplied with the emission probability of the word and store that into viterbi table.
  - Then, By iterating over all other words by calculating transition probabilities multiplied with the product of previous viterbi probability stored in that table and emission probability of that word. At each step for each label we store only the max probability obatined by computing the previous equation. Along with that, we also store the label from which this is transitioned.
  - We continue this step till the end of the sentence.
  - From the max probability obtained for the label at the last word we backtrack and get all the labels from where we transitioned and get to the start word and store all the labels into a list.

- **MCMC Gibbs Sampling**

  - To solve Complicated model, we are sampling the data using Gibbs sampling method.
  - Initially, I have taken a sample that is generated for simplified model for the sentence.
  - For each word I am sampling them with all distinct pos labels and calculating the posterior probabilities for each word over severla iterations and storing every value in a list.
  - Applied log for the probabilities since it generating lower values.
  - Then For every word in a sentence generated some number of samples and their posterior probabilities. From those we are taking the max posterior probability for every word to refine the label which is the best for the word.
  - Finally, we get a sample for every word which has high posterior probability from those we again pick max occurences of label for particular word and return the list of labels for the sentence.

- Challenges Faced:

  - While training the data initially, I was storing the occurences of word in dictionaries with bucket of values which has part of speech labels at even indices and their count of occurences in odd indices of the bucket to restrict collisions.
  - But it was very to difficult to find the probabilitis when needed iterating over the bucket consumed lot of time. Then thought using dictionary of dictionaries because of the faster lookup instead of using list as a buket tried using dictionary as a bucket which was a great relief.

  - For storing transition probabilities, to store first word label it was very difficult so I have 'start' to the transition of first word in sentence for a label.
