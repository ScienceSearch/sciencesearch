## __Steps to run your own experiment__

### Step 1: Copy Configuration Template: `EXAMPLE_slac_config.json`

Take a look at the example for some guidence on how to alter the file to meet your needs. 
All the steps to take will be defined below.

### Step 2: Set Up Training Directory
Define the training directory path in your configuration file:
```
{
  "training": {
    ...
    "directory": "path/to/your/training/files"
    ...
  }
}
```
### Step 3: Populate Training Directory 

### Step 3A: Populate Training Directory with Existing Files
Add your training files to the directory specified in Step 2.

*Note: to use regular preprocessing methods, run these files through the `Preprocessor` object defined in `preprocessing.py` and run `Preprocessor.process_file(filepath)` on each file*


### Step 3B: Populate Training Directory with a Database and Preprocessing 

1. Place your database file in the private_data/ directory

2. Configure Database Path: define the database path in your configuration

```
{
  "database": "private_data/your_database.db"
}
```

*Note: If you are planning on using existing slac_preprocessing() methods, the database structure must include tables `logbook` and `experiemnts` with matching structure to tables in `simplified_elogs.db`*



### Step 4: Create Training Keywords CSV

This step is __essential__ as hyperparamter optemization requires a gold standard to evaluate alogirthm performance with.

See more on this system's hyperparamter optemization in `hyperparam_methods_documentation.ipynb`

Format: file_name,"keyword1","keyword2","keyword3","keyword4"
*Note: there should not be a headings in the csv*


### Step 5: Configure Keywords File Path

Update your configuration to point to the keywords CSV file:

```
{
  "training": {
    ...
    "directory": "path/to/your/training/files",
    "keywords": "path/to/your/keywords.csv"
    ...
  }
}
```

### Step 6: Configure Pickle File Output

This file will save in your training files directory 

```
{
    "training": {
        ...
        "save_file": "filename"
        ...
    }
}
```

### Step 7: Run Quickstart 

Set config file to your new config's file path

Get extracting!