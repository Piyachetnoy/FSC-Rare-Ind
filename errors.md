Traceback (most recent call last):
  File "C:\Users\katob\Desktop\2025piyachetrepo\FAMNET-rare-ind-counting\scripts\hitl_experiment.py", line 701, in <module>
    main()
  File "C:\Users\katob\Desktop\2025piyachetrepo\FAMNET-rare-ind-counting\scripts\hitl_experiment.py", line 693, in main
    save_results(results, workflow_results, config, args.output_dir)
  File "C:\Users\katob\Desktop\2025piyachetrepo\FAMNET-rare-ind-counting\scripts\hitl_experiment.py", line 550, in save_results
    json.dump(report, f, indent=2)
  File "C:\Users\katob\AppData\Local\Programs\Python\Python310\lib\json\__init__.py", line 179, in dump
    for chunk in iterable:
  File "C:\Users\katob\AppData\Local\Programs\Python\Python310\lib\json\encoder.py", line 431, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "C:\Users\katob\AppData\Local\Programs\Python\Python310\lib\json\encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "C:\Users\katob\AppData\Local\Programs\Python\Python310\lib\json\encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "C:\Users\katob\AppData\Local\Programs\Python\Python310\lib\json\encoder.py", line 438, in _iterencode
    o = _default(o)
  File "C:\Users\katob\AppData\Local\Programs\Python\Python310\lib\json\encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type float32 is not JSON serializable