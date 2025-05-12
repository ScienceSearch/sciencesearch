"""
Training pipeline
"""

# TODO: Develop a Notebook supported by this file to demonstrate
# TODO: the keywords in the context of search.
# A. Hyperparameter -> Ensemble
# 1) load list of filenames, user-keywords from a file
#    also in file: epsilon for "best" F1 score,
#    max #algorithms total in ensemble.
# 2) for each pair, run Hyper with the keywords as ground truth
#    get a list of algorithms/parameters with an F1 score
#    within epsilon of the best (Hyper.get_top_f1())
#    and add to the master list
# 3) After all training is run, sort master list by F1 again
#    then take 'max #algorithms' from the top and use that
#    to create the model.Ensemble (via hyper.algorithms_from_results)
# 4) save the ensemble to a file with pickle e.g. ensemble1.pkl
# B. Ensemble -> Search db (example search, not smart or good)
# 1) config file with pickled Ensemble and output 'database' and directory -> Loader
# 2) Loader.add_text() function that runs the ensemble to extract
#    keywords, then puts the resulting text in a directory and the keywords in the 'database'
# C. Search
# 1) A simple class SearchEngine that takes the same config to know the 'database' and directory
# 2) User can call SearchEngine.search(keywords) to retrieve all texts that have one or
#    more keyword matches from the 'database'
