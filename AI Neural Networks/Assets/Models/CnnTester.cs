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

    public Text text;

    public void AnalyseImage()
    {
        runtimeCNN = ModelLoader.Load(modelAsset);

        var worker = WorkerFactory.CreateWorker(runtimeCNN);

        Tensor input = new Tensor(catTestSprite);

        worker.Execute(input);

        string[] outputNames = runtimeCNN.outputs.ToArray();
        Tensor output = worker.PeekOutput();

        List<float> temp = output.ToReadOnlyArray().ToList();
        float answer = temp.Max();

        //Debug.LogFormat("Hey that's a {0}", identifiers[temp.IndexOf(answer)]);
        text.text = string.Format("Hey that's a {0}", identifiers[temp.IndexOf(answer)]);

        input.Dispose();
        output.Dispose();
    }
}
