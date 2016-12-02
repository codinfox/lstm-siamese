// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>

@interface RunModelViewController : UIViewController{
AVAudioRecorder *audioRecorder;
AVAudioPlayer* audioPlayer;

int recordEncoding;
enum
{
    ENC_AAC = 1,
    ENC_ALAC = 2,
    ENC_IMA4 = 3,
    ENC_ILBC = 4,
    ENC_ULAW = 5,
    ENC_PCM = 6,
} encodingTypes;
}

- (IBAction)getUrl:(id)sender;
- (IBAction)record_audio:(id)sender;
- (IBAction)stop_record_and_play:(id)sender;

@property (weak, nonatomic) IBOutlet UITextView *urlContentTextView;
@property (weak, nonatomic) IBOutlet UITextField *urlTextField;

@end
