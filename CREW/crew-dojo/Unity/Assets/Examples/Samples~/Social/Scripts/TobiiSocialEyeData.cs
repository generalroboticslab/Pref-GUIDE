using UnityEngine;


namespace Tobii.Gaming.Examples.Social
{
    public class TobiiSocialEyeData : MonoBehaviour
    {
        private Vector3 _worldGazePoint; // The position in world space where the player is looking.
        private const float _targetDistance = 2f; // A fixed convergence distance of 2m.
        private const float _invalidGazeRayTimer = 2f; // After the gaze ray is invalid for too long, the ray should return to the camera forward direction.

        // 1 Euro Filter which is used to smooth the gaze direction.
        private readonly OneEuroFilter<Vector3> _oneEuroFilter = new OneEuroFilter<Vector3>(90f);
        private float _mincutoff = 7.5f;
        private float _beta = 0.001f; //TODO: test more values
        private float _dcutoff = 1.0f;
        private float _frequency = 90;
        
        private int _lastFrameNumber;
        private float _lastValidGazeRayCounter;

        private void Start()
        {
            // Set the parameters for the 1 Euro Filter. 90hz is chosen but it will automatically update with the correct frequency at runtime.
            _oneEuroFilter.UpdateParams(_frequency, _mincutoff, _beta, _dcutoff);
        }

        public Vector3 GetWorldGazePoint()
        {
            // If the gaze ray is invalid in the current frame, use the previously set world gaze point,
            // but if too much time has passed, return the gaze ray to the forward direction.
            if (!TobiiAPI.GetGazePoint().IsValid)
            {
                _lastValidGazeRayCounter += Time.deltaTime;
                if (_lastValidGazeRayCounter >= _invalidGazeRayTimer)
                {
                    // Determine a filtered world space position in the forward direction of the camera
                    var mainCamera = Camera.main.transform;
                    _worldGazePoint = FilteredWorldGazePoint(mainCamera.position, mainCamera.forward, _targetDistance);
                }
                return _worldGazePoint;
            }

            // Get the latest gaze ray from the eye tracker.
            var gazeRay = Camera.main.ViewportPointToRay(new Vector2(Mathf.Clamp01(TobiiAPI.GetGazePoint().Viewport.x),Mathf.Clamp01(TobiiAPI.GetGazePoint().Viewport.y)));
            
            _lastValidGazeRayCounter = 0;

            // Check if the frame has changed since the last call of this method, in order to avoid filtering duplicate data.
            if (_lastFrameNumber != Time.frameCount)
            {
                // Determine a filtered world space position where the person is looking.
                _worldGazePoint = FilteredWorldGazePoint(gazeRay.origin, gazeRay.direction, _targetDistance);

                _lastFrameNumber = Time.frameCount;
            }
            
            return _worldGazePoint;
        }

        public bool IsLeftEyeBlinking()
        {
            // Get the latest left blink signal from the eye tracker.
            return TobiiAPI.GetUserPresence().IsUserPresent() && !TobiiAPI.GetGazePoint().IsRecent(0.15f);
        }
        
        public bool IsRightEyeBlinking()
        {
            // Get the latest right blink signal from the eye tracker.
            return TobiiAPI.GetUserPresence().IsUserPresent() && !TobiiAPI.GetGazePoint().IsRecent(0.15f);
        }

        private Vector3 FilteredWorldGazePoint(Vector3 rayOrigin, Vector3 rayDirection, float targetDistance)
        {
            // Determine a world space position where the person is looking.
            var target = rayOrigin + rayDirection * targetDistance;
                
            // Apply the 1 Euro Filter to get a smoothed world gaze point.
            return _oneEuroFilter.Filter(target);
        }
    }
}