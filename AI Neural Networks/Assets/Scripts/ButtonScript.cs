using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;
using UnityEditor.UI;
using UnityEngine.UI;
using System;
using SimpleFileBrowser;
using System.IO;

public class ButtonScript : MonoBehaviour
{
    string path = "";
    public RawImage rawImage = null;
    public RawImage ResultingImage = null;
    public Texture2D myTexture = null;
    public Texture2D myResultingTexture = null;
    public Text errorText;

    public GameObject cnnTester;

    public int resWidth = 2550;
    public int resHeight = 3300;

    // Start is called before the first frame update
    void Start()
    {
        FileBrowser.SetFilters(true, new FileBrowser.Filter("Images", ".jpg", ".png"));
        FileBrowser.SetDefaultFilter(".jpg");

        errorText.gameObject.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        myResultingTexture = cnnTester.GetComponent<CnnTester>().finalTexture;
    }

    public void OnStartClick()
    {
        errorText.gameObject.SetActive(false);

        //path = EditorUtility.OpenFilePanel("Show all images (.png) ", "", "png");

        StartCoroutine(ShowLoadDialogCoroutine());
    }

    public void OnLoadClick()
    {
        if (GetTexture() == null)
        {
            errorText.text = "Error(Code: 44712): Select Image First";
            errorText.gameObject.SetActive(true);
        }
        else
        {
            cnnTester.GetComponent<CnnTester>().AnalyseImage(GetTexture());
        }
    }

    public void OnExportClick()
    {
        if (myResultingTexture == null)
        {
            errorText.text = "Error(Code:71424): No resulting image to export.";
            errorText.gameObject.SetActive(true);
        }
        else
        {
            ExportPicture(myResultingTexture);
        }
    }

    IEnumerator DisplayTexture()
    {
        UnityWebRequest www = UnityWebRequestTexture.GetTexture("file:///" + path);

        yield return www.SendWebRequest();

        if (www.isNetworkError || www.isHttpError)
        {
            Debug.Log(www.error);
        }
        else
        {
            myTexture = ((DownloadHandlerTexture)www.downloadHandler).texture;
            rawImage.texture = myTexture;
        }
    }

    IEnumerator ShowLoadDialogCoroutine()
    {
        // Show a load file dialog and wait for a response from user
        // Load file/folder: both, Allow multiple selection: true
        // Initial path: default (Documents), Initial filename: empty
        // Title: "Load File", Submit button text: "Load"
        yield return FileBrowser.WaitForLoadDialog(FileBrowser.PickMode.FilesAndFolders, true, null, null, "Load Files and Folders", "Load");

        // Dialog is closed
        // Print whether the user has selected some files/folders or cancelled the operation (FileBrowser.Success)
        Debug.Log(FileBrowser.Success);

        if (FileBrowser.Success)
        {
            // Print paths of the selected files (FileBrowser.Result) (null, if FileBrowser.Success is false)
            for (int i = 0; i < FileBrowser.Result.Length; i++)
                Debug.Log(FileBrowser.Result[i]);

            // Or, copy the first file to persistentDataPath
            path = Path.Combine(Application.persistentDataPath, FileBrowserHelpers.GetFilename(FileBrowser.Result[0]));
            FileBrowserHelpers.CopyFile(FileBrowser.Result[0], path);

            StartCoroutine(DisplayTexture());
        }
    }

    public Texture2D GetTexture()
    {
        return myTexture;
    }

    public void ExportPicture(Texture2D t)
    {
        Texture2D exTexture = new Texture2D(t.width, t.height, TextureFormat.RGBA32, false);
        exTexture.SetPixels(t.GetPixels());
        string fileName = TextureName(resWidth, resHeight);
        //string exportName = "Results" + "\\" + fileName;
        System.IO.File.WriteAllBytes(fileName, exTexture.EncodeToPNG());
    }

    public static string TextureName(int width, int height)
    {
        return string.Format("{0}/texture_{1}x{2}_{3}.png",
                             Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                             width, height,
                             System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }
}
