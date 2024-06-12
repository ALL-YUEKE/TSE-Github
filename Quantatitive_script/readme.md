Here is the script we do the quantative_analysis in Github Impersonation study.

get_repo.py collects the top 1000 repository from the random keyword.

contributor_event.py collects all commit the contributor from the targeted repository has made.

commit_repo.py collects all commit from the targeted repository.

pipeline is the shell script includes the process of Crosscheck to find the user with multiple email commit. The input of the pipeline is the commit from the targeted repository and commit these contributors made in target repository. The pipeline script includes 8 python scripts (1. checkleak.py 2. leak_history.py 3. filtersus1.py 4. get_contrirepo.py 5. get_concommit.py 6. filterverify.py 7. filteruser.py 8. filtersus2.py) which analyze the original file and generate the processing commit file in the middle of Crosscheck.

user_ratio.py calculates the number of user get involved in multi-email commit compared to the user we collected overall. 
