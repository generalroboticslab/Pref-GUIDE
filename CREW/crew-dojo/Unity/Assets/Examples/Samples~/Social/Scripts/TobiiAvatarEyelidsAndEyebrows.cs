using UnityEngine;

namespace Tobii.Gaming.Examples.Social
{
    public class TobiiAvatarEyelidsAndEyebrows : MonoBehaviour
    {
#pragma warning disable 649 // Field is never assigned
        [Header("Script/Component References")]
        [SerializeField, Tooltip("Eye tracking data source.")] 
        private TobiiSocialEyeData socialEyeData;
        [SerializeField, Tooltip("The face's SkinnedMeshRenderer which contains blendshapes.")]
        private SkinnedMeshRenderer face;
        [SerializeField, Tooltip("The left or right eye transform which is set by TobiiSocialEyeData. This is used to get the vertical gaze angle.")]
        private Transform eye;

        [Header("Settings")]
        [SerializeField, Tooltip("Movement curve of the blendshapes, which is used to set the acceleration/deceleration.")]
        private AnimationCurve blendShapeMovementCurve;
        [SerializeField, Tooltip("The speed of the eyelid in units per second when going from open to closed, or closed to open.")]
        private float eyelidSpeed;
        [SerializeField, Tooltip("The upward angle of the eye at which blendshapes start to have an effect.")]
        private float lookUpBlendShapeStartAngle;
        [SerializeField, Tooltip("The upward angle of the eye at which blendshapes have reached their maximum effect.")]
        private float lookUpBlendShapeEndAngle;
        [SerializeField, Tooltip("The downward angle of the eye at which blendshapes start to have an effect.")]
        private float lookDownBlendShapeStartAngle;
        [SerializeField, Tooltip("The downward angle of the eye at which blendshapes have reached their maximum effect.")]
        private float lookDownBlendShapeEndAngle;
        
#pragma warning restore 649

        private float _leftBlinkCurrentValue;
        private float _rightBlinkCurrentValue;

        private enum BlendShape
        {
            LeftEyeLid = 17,
            RightEyeLid = 18,
            BrowInnerUp = 21,
            BrowOuterUpLeft = 22,
            BrowOuterUpRight = 23,
            EyesLookDown = 62,
            EyesLookUp = 61,
        }

        private const float EyeBrowBlendShapeFactor = 0.25f;

        private void Update()
        {
            // Update the blink blendshapes using the latest eye data from the TobiiSocialEyeData script.
            // The data source can be changed for non-local players, to receive the blink bools and world gaze point over the network.
            UpdateBlinkBlendshape(socialEyeData.IsLeftEyeBlinking(), ref _leftBlinkCurrentValue, BlendShape.LeftEyeLid);
            UpdateBlinkBlendshape(socialEyeData.IsRightEyeBlinking(), ref _rightBlinkCurrentValue, BlendShape.RightEyeLid);

            // Get the vertical gaze angle of the eyes.
            var verticalGazeAngle = eye.transform.localEulerAngles.x;
            verticalGazeAngle = (verticalGazeAngle > 180) ? verticalGazeAngle - 360 : verticalGazeAngle;
            verticalGazeAngle *= -1;
            
            // Update the lookUp, lookDown, and eyebrow blendshapes using the latest eye data from the TobiiSocialEyeData script.
            UpdateLookingUpBlendShapes(socialEyeData.GetWorldGazePoint(), verticalGazeAngle);
            UpdateLookingDownBlendShapes(socialEyeData.GetWorldGazePoint(), verticalGazeAngle);
        }

        private void UpdateBlinkBlendshape(bool isEyeBlinking, ref float currentValue, BlendShape blendShape)
        {
            // If the blink's current value has already reached its target value, there's no need to set the blendshape.
            var target = isEyeBlinking ? 1f : 0f;
            if (Mathf.Approximately(currentValue, target)) return;

            // Increase the current value towards the target value.
            var direction = currentValue < target ? eyelidSpeed : -eyelidSpeed;
            currentValue = Mathf.Clamp01(currentValue + direction * Time.deltaTime);
            
            // Set the blink blendshape for this eye, using the animation curve to smoothly ease in and out (accelerate and decelerate).
            SetBlendShape(blendShape, blendShapeMovementCurve.Evaluate(currentValue));
        }
        
        private void UpdateLookingUpBlendShapes(Vector3 worldGazePoint, float verticalGazeAngle)
        {
            // Calculate a 'look up' gaze angle value on a scale from 0 to 1, from lookUpBlendShapeStartAngle to lookUpBlendShapeEndAngle.
            var lookUpNormalizedValue = Mathf.Clamp01((verticalGazeAngle - lookUpBlendShapeStartAngle) / (lookUpBlendShapeEndAngle - lookUpBlendShapeStartAngle));

            // Slowly increase the effect of the blendShape as the eye move past the start angle. 
            var lookUpBlendShapeValue = blendShapeMovementCurve.Evaluate(lookUpNormalizedValue);

            // Set the 'look up' blendshape values. Brows shouldn't move much when you look up, so the value is decreased with EyeBrowBlendShapeFactor.
            SetBlendShape(BlendShape.EyesLookUp, lookUpBlendShapeValue);
            SetBlendShape(BlendShape.BrowInnerUp, lookUpBlendShapeValue * EyeBrowBlendShapeFactor);
            SetBlendShape(BlendShape.BrowOuterUpLeft, lookUpBlendShapeValue * EyeBrowBlendShapeFactor);
            SetBlendShape(BlendShape.BrowOuterUpRight, lookUpBlendShapeValue * EyeBrowBlendShapeFactor);
        }

        private void UpdateLookingDownBlendShapes(Vector3 worldGazePoint, float verticalGazeAngle)
        {
            // Calculate a 'look down' gaze angle value on a scale from 0 to 1, from lookDownBlendShapeStartAngle to lookDownBlendShapeEndAngle.
            var lookDownNormalizedValue = Mathf.Clamp01((verticalGazeAngle - lookDownBlendShapeStartAngle) / (lookDownBlendShapeEndAngle - lookDownBlendShapeStartAngle));

            // When blinking while looking down, eyesLookDown and eyeBlink act at the same time, unrealistically stretching down the top eyelid.
            // To avoid this, the eyelids are only moved down by the amount not used by the blink blendShapes.
            var blinkValue = Mathf.Max(_leftBlinkCurrentValue, _rightBlinkCurrentValue);
            lookDownNormalizedValue = Mathf.Clamp(lookDownNormalizedValue, 0, 1 - blinkValue);

            // Slowly increase the effect of the blendShape as the eye move past the start angle. 
            var lookDownBlendShapeValue = blendShapeMovementCurve.Evaluate(lookDownNormalizedValue);

            // Set the 'look down' blendshape value.
            SetBlendShape(BlendShape.EyesLookDown, lookDownBlendShapeValue);
        }
        
        private void SetBlendShape(BlendShape blendShape, float normalizedValue)
        {
            // Set the desired blendshape, using a normalizedValue (0 to 1) and multiplying by 100 to fit the SkinnedMeshRenderer blendshape value range (0 to 100).
            face.SetBlendShapeWeight((int) blendShape, normalizedValue * 100f);
        }
    }
}