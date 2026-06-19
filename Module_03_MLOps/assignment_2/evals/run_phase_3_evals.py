import requests
import json

TAGS = {"phase": "4", "run": "manual_trace_test"}

# Define the 5 test cases from your eval set
tests = [
    {
        "db": "formula_1", 
        "question": "What is the average fastest lap time in seconds for Lewis Hamilton in all the Formula_1 races?"
    },
    {
        "db": "formula_1", 
        "question": "What is the coordinates location of the circuits for Australian grand prix?"
    },
    {
        "db": "superhero", 
        "question": "List down Ajax's superpowers."
    },
    {
        "db": "california_schools", 
        "question": "List the top five schools, by descending order, from the highest to the lowest, the most number of Enrollment (Ages 5-17). Please give their NCES school identification number."
    },
    {
        "db": "financial", 
        "question": "How many male clients in 'Hl.m. Praha' district?"
    },
    {
        "db": "student_club",
        "question": "In the College of Agriculture and Applied Sciences, how many majors are under the department of School of Applied Sciences, Technology and Education?"
    },
    {
        "db": "codebase_community",
        "question": "Mention the display name and location of the user who owned the excerpt post with hypothesis-testing tag."
    },
    {
        "db": "thrombosis_prediction",
        "question": "List all patients who were followed up at the outpatient clinic who underwent a laboratory test in October 1991 and had a total blood bilirubin level within the normal range."
    },
    {
        "db": "toxicology",
        "question": "Among the molecules with element Calcium, are they mostly carcinogenic or non carcinogenic?"
    },
    {
        "db": "student_club",
        "question": "Please list the full names of the students in the Student_Club that come from the Art and Design Department."
    }
]

# The local tunnel endpoint
URL = "http://localhost:9001/answer"

for i, test in enumerate(tests, 1):
    print(f"\n{'='*80}")
    print(f"📝 TEST {i}/5 | Database: {test['db']}")
    print(f"❓ Question: {test['question']}")
    print(f"{'-'*80}")
    
    try:
        # Send the POST request
        response = requests.post(URL, json={**test, "tags": TAGS})
        response.raise_for_status()
        data = response.json()
        
        print(f"🔄 Total Iterations required: {data.get('iterations', 'Unknown')}\n")
        print("🔍 AGENT THOUGHT PROCESS:")
        
        # Loop through the execution history to show each step clearly
        for step, action in enumerate(data.get('history', []), 1):
            node = action.get('node', 'unknown')
            
            if node == 'generate_sql':
                print(f"  [{step}] 🛠️  GENERATE: Drafted initial SQL.")
                print(f"       Query: {action.get('sql').strip().replace(chr(10), ' ')}")
                
            elif node == 'verify':
                passed = action.get('ok')
                status = "✅ PASSED" if passed else "❌ FAILED"
                print(f"  [{step}] ⚖️  VERIFY: {status}")
                if not passed:
                    print(f"       Issue spotted: {action.get('issue')}")
                    
            elif node == 'revise':
                print(f"  [{step}] 🔧 REVISE: Rewrote SQL based on verify feedback.")
                print(f"       Query: {action.get('sql').strip().replace(chr(10), ' ')}")
                
        print(f"\n📊 FINAL RESULT:")
        if data.get('error'):
            print(f"   ⚠️ Error: {data.get('error')}")
        else:
            print(f"   ✅ Rows Returned: {data.get('rows')}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: Ensure your tunnel (port 9001) and server are running. Details: {e}")