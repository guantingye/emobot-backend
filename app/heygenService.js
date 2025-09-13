// src/services/heygenService.js
class HeyGenService {
    constructor() {
      this.apiKey = import.meta.env.VITE_HEYGEN_API_KEY || '';
      this.avatarId = import.meta.env.VITE_HEYGEN_AVATAR_ID || 'default_avatar_id';
      this.sessionInfo = null;
      this.peerConnection = null;
      this.mediaStream = null;
    }
  
    // 初始化 WebRTC 連接
    async initializeWebRTC(sessionInfo) {
      const pc = new RTCPeerConnection({ 
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] 
      });
  
      // 設定遠端描述
      await pc.setRemoteDescription(new RTCSessionDescription({
        type: 'offer',
        sdp: sessionInfo.sdp
      }));
  
      // 創建回應
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
  
      // 監聽 ICE 候選
      pc.onicecandidate = (event) => {
        if (event.candidate) {
          this.sendICECandidate(event.candidate);
        }
      };
  
      // 監聽媒體流
      pc.ontrack = (event) => {
        if (event.streams && event.streams[0]) {
          this.mediaStream = event.streams[0];
        }
      };
  
      this.peerConnection = pc;
      return answer;
    }
  
    // 建立串流會話
    async createStreamingSession() {
      try {
        const response = await fetch('https://api.heygen.com/v2/streaming/new', {
          method: 'POST',
          headers: {
            'X-Api-Key': this.apiKey,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            avatar_id: this.avatarId,
            voice: {
              voice_id: 'zh-TW-HsiaoChenNeural', // 繁體中文語音
              rate: 1.0,
              pitch: 0
            }
          })
        });
  
        if (!response.ok) {
          throw new Error(`HeyGen API error: ${response.status}`);
        }
  
        const data = await response.json();
        this.sessionInfo = data.data;
  
        // 初始化 WebRTC
        const answer = await this.initializeWebRTC(this.sessionInfo);
  
        // 啟動會話
        await this.startSession(answer);
  
        return this.sessionInfo.session_id;
      } catch (error) {
        console.error('Failed to create streaming session:', error);
        throw error;
      }
    }
  
    // 啟動會話
    async startSession(answer) {
      try {
        const response = await fetch('https://api.heygen.com/v2/streaming/start', {
          method: 'POST',
          headers: {
            'X-Api-Key': this.apiKey,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            session_id: this.sessionInfo.session_id,
            sdp: {
              type: 'answer',
              sdp: answer.sdp
            }
          })
        });
  
        if (!response.ok) {
          throw new Error(`Failed to start session: ${response.status}`);
        }
      } catch (error) {
        console.error('Failed to start session:', error);
        throw error;
      }
    }
  
    // 發送 ICE 候選
    async sendICECandidate(candidate) {
      try {
        await fetch('https://api.heygen.com/v2/streaming/ice', {
          method: 'POST',
          headers: {
            'X-Api-Key': this.apiKey,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            session_id: this.sessionInfo.session_id,
            candidate: {
              candidate: candidate.candidate,
              sdpMLineIndex: candidate.sdpMLineIndex,
              sdpMid: candidate.sdpMid,
              usernameFragment: candidate.usernameFragment
            }
          })
        });
      } catch (error) {
        console.error('Failed to send ICE candidate:', error);
      }
    }
  
    // 發送文字讓 Avatar 說話
    async sendTextToSpeak(text) {
      if (!this.sessionInfo || !this.sessionInfo.session_id) {
        throw new Error('No active session');
      }
  
      try {
        const response = await fetch('https://api.heygen.com/v2/streaming/task', {
          method: 'POST',
          headers: {
            'X-Api-Key': this.apiKey,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            session_id: this.sessionInfo.session_id,
            task: {
              type: 'repeat',
              text: text
            }
          })
        });
  
        if (!response.ok) {
          throw new Error(`Failed to send text: ${response.status}`);
        }
  
        const data = await response.json();
        return data.data.task_id;
      } catch (error) {
        console.error('Failed to send text to speak:', error);
        throw error;
      }
    }
  
    // 取得媒體流
    getMediaStream() {
      return this.mediaStream;
    }
  
    // 結束會話
    async endSession() {
      if (!this.sessionInfo || !this.sessionInfo.session_id) {
        return;
      }
  
      try {
        await fetch('https://api.heygen.com/v2/streaming/stop', {
          method: 'POST',
          headers: {
            'X-Api-Key': this.apiKey,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            session_id: this.sessionInfo.session_id
          })
        });
      } catch (error) {
        console.error('Failed to end session:', error);
      }
  
      // 清理資源
      if (this.peerConnection) {
        this.peerConnection.close();
        this.peerConnection = null;
      }
      
      this.sessionInfo = null;
      this.mediaStream = null;
    }
  }
  
  export default new HeyGenService();