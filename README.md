## Isomer Recommender Training Module

This repo contains the training module for the Isomer recommender system. The Isomer recommender system uses the [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) algorithm to predict related pages on government websites using the Isomer platform.

This README is meant for the **Admins** of Isomer. It covers information on how to set up and edit the training module. To find out more about how to set up the recommender on individual Isomer websites, please refer to the [Isomer recommender-predict](https://github.com/isomerpages/recommender-predict) repository.

### How to add the recommender feature to new Isomer sites

`isomer-sites.json` is the configuration file that provides the main `application.py` script with information on which Isomer repo to run the recommender training on. 

The Isomer recommender supports two recommendation modes: intra-site and inter-site.

**Intra-site recommendations: recommend pages within the same site**
To add intra-site recommender predictions to a new Isomer site, add a new array containing a single JSON object with the following fields: `git_url` and `site_url`. Refer to `"0"` in the example.

**Inter-site recommendations: recommend pages across different sites**
To add inter-site recommender predictions for two or more Isomer sites, add a new array containing multiple JSON objects with `git_url` and `site_url` to the `isomer-sites.json` file. Refer to `"1"` in the example.

```
# Sample isomer-sites.json file
# Example 0: Intra-site recommendations
# Example 1: Inter-site recommendations

{
  "0": [
    {
      "git_url": "https://github.com/isomerpages/isomerpages-govtech.git",
      "site_url": "https://www.tech.gov.sg"
    }
  ],
  "1": [
    {
      "git_url": "https://github.com/isomerpages/isomerpages-boa.git",
      "site_url": "https://www.boa.gov.sg"
    }, {
      "git_url": "https://github.com/isomerpages/isomerpages-hlb.git",
      "site_url": "https://www.hlb.gov.sg"
    }
  ]
}
```

### How to develop locally

**Install dependencies and set up environment variables**
```
source .env
pip install -r requirements.txt
```

**Run `application.py`**

```
python3 application.py
```

### How to set up on TravisCI

**Set the various environment variables on the Travis UI**

The variables are `AWS_REGION_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DYNAMODB_ENDPOINT`, `AWS_DYNAMODB_TABLE_NAME`.

A cron job on TravisCI will run once every day.