Here is the script setup we use for the quantitative analysis in our GitHub Impersonation Study:

1. **get_repo.py**: This script collects the top 1000 repositories based on random keywords.  
   The output is the targeted repositories for this study.

2. **contributor_event.py**: This script gathers all commits made by contributors from the targeted repositories.  
   The output is stored in the "contributor_commit" directory within Quantatitive_data.

3. **commit_repo.py**: This script retrieves all commits from the targeted repositories.  
   The output is stored in the "repository_commit" directory within Quantatitive_data.

Both the "repository_commit" and "contributor_commit" directories serve as the basis for further analysis. 

4. **pipeline**: This shell script includes processes for Crosscheck to identify users with multiple email commits.  
   It takes the commits from the targeted repositories and the commits made by these contributors as inputs. The pipeline script consists of 8 Python scripts which analyze the original files and generate processed commit files during the Crosscheck process:
   - checkleak.py
   - leak_history.py
   - filtersus1.py
   - get_contrirepo.py
   - get_concommit.py
   - filterverify.py
   - filteruser.py
   - filtersus2.py

The result from the pipeline identifies users involved in multi-email commits, a tactic similar to impersonation. The result is also released in Quantatitive_data as cross_check_result.

5. **user_ratio.py**: This script calculates the ratio of users involved in multi-email commits compared to the overall number of collected users.

6. **fullcheck.py**: This script checks whether the collected commits are signing commits.

7. **countsign.py**: This script calculates the number of signing commits and the number of non-signing commits.
