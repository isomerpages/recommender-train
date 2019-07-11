## Isomer Recommender Training Module

This repo contains the training module for the Isomer recommender system. The Isomer recommender system uses the [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) algorithm to predict related pages on government websites using the Isomer platform.

### How to run locally

**Install dependencies and set up environment variables**
```
source .env
pip install -r requirements.txt
```

**Update `isomer-sites.json` with the relevant site details**

This configuration json file provides the main `application.py` script with information on which Isomer repo to run the recommender training on.

Note: at this point, the recommender system works intra-site. That is, it generates page recommendations within an Isomer repo. We plan to expand the recommender to perform cross-site recommendations in the near future.

```
# isomer-sites.json

[
  {
    "git_url": "https://github.com/isomerpages/isomerpages-govtech.git",
    "site_url": "https://www.tech.gov.sg",
    "directory_name": "./tmp/govtech"
  }
]
```

**Run `application.py`**

```
python3 application.py
```

### How to run on TravisCI

**Set the various environment variables on the Travis UI**
The variables are `AWS_REGION_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DYNAMODB_ENDPOINT`, `AWS_DYNAMODB_TABLE_NAME`.

A cron job on TravisCI will run once every day.