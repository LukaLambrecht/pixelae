# Alternative way of retreiving input data using DIALS

Status: currently does not seem to work well yet,
jobs go into error with following message:
```
INFO: Api raw response: <html>
<head><title>504 Gateway Time-out</title></head>
<body>
<center><h1>504 Gateway Time-out</h1></center>
<hr><center>nginx/1.25.5</center>
</body>
</html>
```
followed by a JSON decoder error.

Maybe need to split calls into smaller chunks, e.g. first retrieve all run numbers
in a dataset and then make a separate call per run number.
This has now been implemented but does not seem to help, still same error.
