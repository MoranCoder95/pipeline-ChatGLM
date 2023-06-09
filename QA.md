# QA

1 ERROR - pipelines.nodes.file_converter.pdf -  File ./data/1.pdf has an error 
 Unable to get page count. Is poppler installed and in PATH?
🎉 PDFToTextOCRConverter()使用中需要poppler支持，windows安装可见链接 [Poppler for Windows](https://blog.alivate.com.au/poppler-windows/)和[安装教程](https://blog.csdn.net/wy01415/article/details/110257130)

2 RuntimeError: Library cublasLt is not initialized
🎉显存问题，更换更大显存或者报错后再运行一次看看能不能正常运行

