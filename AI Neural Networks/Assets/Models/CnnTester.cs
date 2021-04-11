using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

public class CnnTester : MonoBehaviour
{
    public Texture2D catTestSprite;
    public NNModel modelAsset;
    private Model runtimeCNN;

    private string[] identifiers = new string[] { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

    public Image outputImage;
    public Text text;
    public RenderTexture outputTexture;
    public Texture2D finalTexture;
    public Texture2D alteredTexture;

    public void AnalyseImage()
    {
        runtimeCNN = ModelLoader.Load(modelAsset);

        var worker = WorkerFactory.CreateWorker(runtimeCNN);

        Tensor input = new Tensor(catTestSprite, 3);

        worker.Execute(input);

        string[] outputNames = runtimeCNN.outputs.ToArray();
        Tensor output = worker.PeekOutput();

        /* Old Image Classification Code
        List<float> temp = output.ToReadOnlyArray().ToList();
        float answer = temp.Max();
        //Debug.LogFormat("Hey that's a {0}", identifiers[temp.IndexOf(answer)]);
        text.text = string.Format("Hey that's a {0}", identifiers[temp.IndexOf(answer)]);
        */

        outputTexture = new RenderTexture(1024, 1024, 3);
        BarracudaTextureUtils.TensorToRenderTexture(output, outputTexture);
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = outputTexture;
        finalTexture = new Texture2D(outputTexture.width, outputTexture.height, TextureFormat.RGBA32, false);
        finalTexture.ReadPixels(new Rect(0, 0, outputTexture.width, outputTexture.height), 0, 0, false);
        finalTexture.Apply();
        
        RenderTexture.active = currentRT;
        //outputImage.sprite = Sprite.Create(finalTexture, new Rect(0, 0, finalTexture.width, finalTexture.height), new Vector2(0.5f, 0.5f));
        outputImage.sprite = Sprite.Create(finalTexture, new Rect(0, 0, finalTexture.width, finalTexture.height), new Vector2(0.5f, 0.5f));

        input.Dispose();
        output.Dispose();
    }
}
