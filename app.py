from flask import Flask, render_template, request, url_for
import os

app = Flask(__name__)

@app.route("/")
def main():
	return render_template("index.html")


@app.route("/summarize",methods=['POST'])
def summarize():
	_text = request.form['input_text']
	if len(_text) > 0:
	 	f = open("input_file.txt","w+")
	 	print(_text)
	 	f.write(_text)
	 	f.close()

	 	os.system("python3.6 present.py")

	 	f = open("output_file.txt","r+")	
	 	txt = f.read()
	 	f.close()
	 	result = [_text,txt]	
	return render_template("summarize.html",result=result)		

if __name__ == "__main__":
	app.run()	
