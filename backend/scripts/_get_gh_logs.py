import urllib.request, json
try:
    url = "https://api.github.com/repos/SKYLINE217/On-Chain-Fraud-Detection-System/actions/runs?per_page=1"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        runs = json.loads(response.read().decode())
    
    if not runs['workflow_runs']:
        print("No runs found")
    else:
        run_id = runs['workflow_runs'][0]['id']
        print(f"Latest Run ID: {run_id}")
        
        jobs_url = f"https://api.github.com/repos/SKYLINE217/On-Chain-Fraud-Detection-System/actions/runs/{run_id}/jobs"
        req = urllib.request.Request(jobs_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            jobs = json.loads(response.read().decode())
            
        for job in jobs['jobs']:
            if job['conclusion'] == 'failure':
                print(f"Job failed: {job['name']}")
                for step in job['steps']:
                    if step['conclusion'] == 'failure':
                        print(f"Step failed: {step['name']}")
                        # Unfortunately, logs require authentication for public repos via API sometimes, but let's see
except Exception as e:
    print(f"Error: {e}")
