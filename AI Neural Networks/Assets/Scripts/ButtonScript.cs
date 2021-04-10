using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using SimpleFileBrowser;
using System.IO;

public class ButtonScript : MonoBehaviour
{
    string path = "";
    public RawImage rawImage;
    public RawImage rawResultingImage;
    public Texture2D myTexture = null;
    public Texture2D myResultingTexture = null;
    public Text errorText;

    public int resWidth = 2550;
    public int resHeight = 3300;

    // Start is called before the first frame update
    void Start()
    {
        errorText.gameObject.SetActive(false);

        FileBrowser.SetFilters(true, new FileBrowser.Filter("Images", ".jpg", ".png"), new FileBrowser.Filter("Text Files", ".txt", ".pdf"));
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void OnButtonClick()
    {
        errorText.gameObject.SetActive(false);

        StartCoroutine(ShowLoadDialogCoroutine());
    }

    public void OnButtonClick2()
    {
        if (GetTexture() == null)
        {
            errorText.gameObject.SetActive(true);
        }
        else
        {
            ExportPicture(myResultingTexture);
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

            StartCoroutine(DisplayTexture());
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
            myResultingTexture = ((DownloadHandlerTexture)www.downloadHandler).texture;
            rawImage.texture = myTexture;
            rawResultingImage.texture = myResultingTexture;
        }
    }

    public Texture GetTexture()
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
        return string.Format("{0}/Results/texture_{1}x{2}_{3}.png",
                             Application.dataPath,
                             width, height,
                             System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }
}
