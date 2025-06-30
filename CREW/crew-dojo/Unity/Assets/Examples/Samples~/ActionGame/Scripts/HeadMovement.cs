//-----------------------------------------------------------------------
// Copyright 2016 Tobii AB (publ). All rights reserved.
//-----------------------------------------------------------------------

using UnityEngine;

namespace Tobii.Gaming.Examples.ActionGame
{
	public class HeadMovement : MonoBehaviour
	{
		public Transform Head;
		public float Responsiveness = 10f;
		public bool HeadPositionEnabled = false;

		private Vector3? startPosition = null;

		void Update()
		{
			var headPose = TobiiAPI.GetHeadPose();
			if (headPose.IsRecent())
			{
				Head.transform.localRotation = Quaternion.Lerp(Head.transform.localRotation, headPose.Rotation,
					Time.unscaledDeltaTime * Responsiveness);

				if (HeadPositionEnabled)
				{
					if (startPosition == null)
					{
						startPosition = headPose.Position;
					}

					var positionOffset = headPose.Position - startPosition.Value;
					Head.transform.localPosition = Vector3.Lerp(Head.transform.localPosition, positionOffset,
						Time.unscaledDeltaTime * Responsiveness);
				}
			}
		}
	}
}
