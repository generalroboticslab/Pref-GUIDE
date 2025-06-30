using UnityEngine;

namespace Tobii.Gaming.Examples.Social
{
    public class TobiiAvatarEyes : MonoBehaviour
    {
#pragma warning disable 649 // Field is never assigned
        [SerializeField, Tooltip("Left eye transform which should face directly forward from the character.")] 
        private Transform leftEye;
        [SerializeField, Tooltip("Right eye transform which should face directly forward from the character.")] 
        private Transform rightEye;
        [SerializeField, Tooltip("Eye tracking data source.")]
        private TobiiSocialEyeData tobiiSocialEyeData;
#pragma warning restore 649

        private void Update()
        {
            // Get the latest eye data from the TobiiSocialEyeData script.
            // The data source can be changed for non-local players, to receive the world gaze point over the network.
            var worldGazePoint = tobiiSocialEyeData.GetWorldGazePoint();
            
            // Rotate each eye to look at the world gaze point.
            leftEye.LookAt(worldGazePoint);
            rightEye.LookAt(worldGazePoint);
        }
    }
}