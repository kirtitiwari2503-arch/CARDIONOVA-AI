from flask import Flask, jsonify
app = Flask(_ _ name_ _)

@app.route("/api/health")
def health():
    return jsonify({"status" :" running"})

if __name__ == "__ main__":
    app.run(debug = True)
